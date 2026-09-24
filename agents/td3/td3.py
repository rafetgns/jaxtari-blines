import os
import random
import time
from functools import partial

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flashbax as fbx
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import jaxatari
from jaxatari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper
)
from agents.td3.td3_eval import evaluate
from rtpt import RTPT

# Gumbell softmax is used to make it compatible with discrete actions because gradients are calculated through samples.
# However, that doesn't provide stable comvergence for DDPG for Atari.

def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False):
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None
    if not eval and mods is not None and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxatari.make(env_id, mods=mods)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,
            first_fire=True,
            noop_max=30,
            full_action_space=False,
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                use_native_downscaling=native_downscaling,
                smooth_image=False,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=not eval,
            )
        else:
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=not eval,
                    )
                )
            )
        env = LogWrapper(env)
        return env
    return thunk

class Pixel_Actor_Discrete(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        return x 

class Pixel_Critic(nn.Module):
    @nn.compact
    def __call__(self, x, a):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate([x,a], axis=-1)
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(1, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        return x


class MLP_Actor_Discrete(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        return x 

class MLP_Critic(nn.Module):
    @nn.compact
    def __call__(self, x, a):
        x = jnp.concatenate([x,a], axis=-1)
        x = nn.Dense(256, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(1, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        return x


class TD3TrainState(TrainState):
    target_params: flax.core.FrozenDict


@flax.struct.dataclass
class TimeStep:
    obs: jnp.array
    action: jnp.array
    reward: jnp.array
    done: jnp.array


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if config.get("PIXEL_BASED", True) and config.get("NUM_ENVS", 1) > 16:
        print("Warning: More than 16 environments may cause OOM on GPU when using pixel-based observations.") 

    run_name = f"{config['ENV_ID']}_{config['EXP_NAME']}_{'oc' if not config['PIXEL_BASED'] else 'pixel'}_{config['SEED']}"

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
    )
    wandb.define_metric("*", step_metric="charts/global_step")

    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    train_mods = list(config.get("TRAIN_MODS", []))

    env = make_env(
        config.get("ENV_ID"),
        train_mods,
        config.get("PIXEL_BASED", True),
        config.get("NATIVE_DOWNSCALING", True),
        False,
    )()

    action_dim = env.action_space().n
    

    obs_shape = env.observation_space().shape
    if config.get("PIXEL_BASED", True):
        obs_shape = obs_shape[:-1]

    num_envs = config["NUM_ENVS"]
    # if -1: we do as many gradient steps as collected samples (stable_baselines3 behavior), gradient_steps = 1  original TD3 paper
    gradient_steps = num_envs * config.get("TRAIN_FREQUENCY", 4) if config.get("GRADIENT_STEPS", 1) == -1 else config.get("GRADIENT_STEPS", 1) 

    @jax.jit
    def vmap_reset(rng):
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(rng.shape[0], *obs_shape), state

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs.reshape(action.shape[0], *obs_shape), state, reward, next_done, info

    gamma = config.get("GAMMA", 0.99)
    tau = config.get("TAU", 1.0)
    batch_size = config.get("BATCH_SIZE", 64)
    policy_update_frequency = config.get("POLICY_UPDATE_FREQUENCY", 2)
    target_update_frequency = config.get("TARGET_UPDATE_FREQUENCY", 8000)
    learning_starts = config.get("LEARNING_STARTS", 20000)
    steps_per_update = config.get("TRAIN_FREQUENCY", 4) * config.get("NUM_ENVS", 1)


    key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)
    
    if config.get("PIXEL_BASED", True):
        actor_net = Pixel_Actor_Discrete(action_dim=action_dim)
        critic_net = Pixel_Critic()
    else:
        actor_net = MLP_Actor_Discrete(action_dim=action_dim)
        critic_net = MLP_Critic()

    dummy_obs = jnp.zeros((1, *obs_shape))
    dummy_act = jnp.zeros((1, action_dim))

    actor_state = TD3TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_net.init(actor_key, dummy_obs),
        target_params=actor_net.init(actor_key, dummy_obs),
        tx=optax.adam(learning_rate=config.get("LEARNING_RATE", 3e-4), eps=1e-4),
    )
    qf1_state = TD3TrainState.create(
        apply_fn=critic_net.apply,
        params=critic_net.init(qf1_key, dummy_obs, dummy_act),
        target_params=critic_net.init(qf1_key, dummy_obs, dummy_act),
        tx=optax.adam(learning_rate=config.get("LEARNING_RATE", 3e-4), eps=1e-4),
    )
    qf2_state = TD3TrainState.create(
        apply_fn=critic_net.apply,
        params=critic_net.init(qf2_key, dummy_obs, dummy_act),
        target_params=critic_net.init(qf2_key, dummy_obs, dummy_act),
        tx=optax.adam(learning_rate=config.get("LEARNING_RATE", 3e-4), eps=1e-4),
    )

    replay_buffer = fbx.make_prioritised_flat_buffer(
        max_length=config.get("BUFFER_SIZE", int(1e6)),
        min_length=learning_starts,
        sample_batch_size=batch_size,
        add_sequences=False,
        add_batch_size=num_envs,
    )
    replay_buffer = replay_buffer.replace(
            init=jax.jit(replay_buffer.init),
            add=jax.jit(replay_buffer.add, donate_argnums=0),
            sample=jax.jit(replay_buffer.sample),
            can_sample=jax.jit(replay_buffer.can_sample),
    )
    _obs, _state = vmap_reset(jax.random.split(key, num_envs))
    _obs, _state, _reward, _done, _info = vmap_step(_state, jnp.zeros((num_envs,), dtype=jnp.int32))
    
    _dummy_step = TimeStep(
        obs=_obs[0],
        action=jnp.zeros((), dtype=jnp.int32),
        reward=_reward[0],
        done=_done[0],
    )
    buffer_state = replay_buffer.init(_dummy_step)

    @jax.jit
    def gumbel_softmax_sample(logits, key, temperature=1.0):

        gumbel_noise = jax.random.gumbel(key, shape=logits.shape, dtype=logits.dtype)    

        y = logits + gumbel_noise
        y_soft = jax.nn.softmax(y / temperature, axis=-1)
        action_idx = jnp.argmax(y_soft, axis=-1)
        y_hard = jax.nn.one_hot(action_idx, num_classes=logits.shape[-1], dtype=logits.dtype)      
        y_hard = jax.lax.stop_gradient(y_hard - y_soft) + y_soft
        
        return y_hard, y_soft, action_idx

    def full_td3_step(actor_state, qf1_state, qf2_state, buffer_state, env_state, obs, rng, global_step, temperature):
        
        def take_action(carry, _):
            actor_state, buffer_state, env_state, obs, global_step, rng = carry
            rng, action_rng, noise_rng = jax.random.split(rng, 3)
            action_sample_keys = jax.random.split(action_rng, num_envs)

            random_actions = jax.vmap(env.action_space().sample)(action_sample_keys)
            det_actions = actor_state.apply_fn(actor_state.params, obs)
            _, _, noisy_action_idx = gumbel_softmax_sample(det_actions, noise_rng, temperature=temperature)

            actions = jnp.where(global_step < learning_starts, random_actions, noisy_action_idx)
            next_obs, next_env_state, rewards, next_done, info = vmap_step(env_state, actions)


            timestep = TimeStep(
                obs=obs,
                action=actions,
                reward=rewards,
                done=next_done,
            )
            buffer_state = replay_buffer.add(buffer_state, timestep)
            return (actor_state, buffer_state, next_env_state, next_obs, global_step + num_envs, rng), info

        (actor_state, buffer_state, next_env_state, next_obs, global_step, rng), infos = jax.lax.scan(
            take_action,
            (actor_state, buffer_state, env_state, obs, global_step, rng), 
            None,
            length=config.get("TRAIN_FREQUENCY", 4),
        )

        def do_update(update_carry, _):
            u_actor_state, u_qf1_state, u_qf2_state, u_key, gradient_step_counter = update_carry
            u_key, sample_key, sample_key2, sample_key3 = jax.random.split(u_key, 4)

            batch = replay_buffer.sample(buffer_state, sample_key).experience
            b_obs = batch.first.obs
            b_act = batch.first.action
            b_rew = batch.first.reward
            b_don = batch.first.done
            b_nobs = batch.second.obs

            #Discrete
            next_actions_hard, _, _ = gumbel_softmax_sample(u_actor_state.apply_fn(u_actor_state.target_params, b_nobs), sample_key2, temperature=temperature)

            q1_next_target = u_qf1_state.apply_fn(u_qf1_state.target_params, b_nobs, next_actions_hard).reshape(-1)
            q2_next_target = u_qf2_state.apply_fn(u_qf2_state.target_params, b_nobs, next_actions_hard).reshape(-1)
            min_q_next_target = jnp.minimum(q1_next_target, q2_next_target)
            next_q_value = (b_rew.flatten() + (1.0 - b_don.flatten()) * gamma * min_q_next_target).reshape(-1)
            next_q_value = jax.lax.stop_gradient(next_q_value)

            def qf_loss_fn(qf_params, qf_state):
                b_act_onehot = jax.nn.one_hot(b_act,num_classes=action_dim)
                qf_pred = qf_state.apply_fn(qf_params, b_obs, b_act_onehot).reshape(-1)
                qf_loss = jnp.mean((qf_pred - next_q_value) ** 2)
                return qf_loss, qf_pred.mean()

            (qf1_loss, qf1_val), grads1 = jax.value_and_grad(qf_loss_fn, has_aux=True)(u_qf1_state.params, u_qf1_state)
            (qf2_loss, qf2_val), grads2 = jax.value_and_grad(qf_loss_fn, has_aux=True)(u_qf2_state.params, u_qf2_state)
            
            new_qf1_state = u_qf1_state.apply_gradients(grads=grads1)
            new_qf2_state = u_qf2_state.apply_gradients(grads=grads2)

            # Delayed Policy Update
            def perform_actor_update(c):
                c_actor, c_qf1 = c
                def actor_loss_fn(actor_params):
                    gumbel_sample , _, _ = gumbel_softmax_sample(c_actor.apply_fn(actor_params, b_obs), sample_key3, temperature=temperature)
                    return -c_qf1.apply_fn(c_qf1.params, b_obs, gumbel_sample).mean()
                
                actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(c_actor.params)
                updated_actor = c_actor.apply_gradients(grads=actor_grads)
        
                return updated_actor, actor_loss
            
            def skip_actor_update(c):
                c_actor, c_qf1 = c
                def actor_loss_fn(actor_params):
                    gumbel_sample , _, _ = gumbel_softmax_sample(c_actor.apply_fn(actor_params, b_obs), sample_key3, temperature=temperature)
                    return -c_qf1.apply_fn(c_qf1.params, b_obs, gumbel_sample).mean()
                return c_actor, actor_loss_fn(c_actor.params)

            u_actor_state, actor_loss = jax.lax.cond(
                gradient_step_counter % policy_update_frequency == 0,
                perform_actor_update,
                skip_actor_update,
                (u_actor_state, new_qf1_state)
            )

            
            return (u_actor_state, new_qf1_state, new_qf2_state, u_key, gradient_step_counter + 1), (qf1_loss, qf2_loss, actor_loss, qf1_val)

        
        def scanned_update(carry):
            carry, metrics = jax.lax.scan(do_update, carry, None, length=gradient_steps)
            qf1_l, qf2_l, act_l, qf1_v = metrics
            return carry, (qf1_l[-1], qf2_l[-1], act_l[-1], qf1_v[-1])

        (actor_state, qf1_state, qf2_state, rng, gradient_step_counter), (qf1_loss, qf2_loss, actor_loss, qf1_val) = jax.lax.cond(
            replay_buffer.can_sample(buffer_state),
            lambda c: scanned_update(c),
            lambda c: (c, (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))),
            (actor_state, qf1_state, qf2_state, rng, global_step // steps_per_update), 
        )

        update_target_flag = jnp.logical_and(
            replay_buffer.can_sample(buffer_state),
            (global_step % target_update_frequency) < steps_per_update
        )
                
        def update_target_networks(c):
            c_actor, c_qf1, c_qf2 = c
            updated_actor = c_actor.replace(
                target_params=optax.incremental_update(c_actor.params, c_actor.target_params, tau)
            )
            updated_qf1 = c_qf1.replace(
                target_params=optax.incremental_update(c_qf1.params, c_qf1.target_params, tau)
            )
            updated_qf2 = c_qf2.replace(
                target_params=optax.incremental_update(c_qf2.params, c_qf2.target_params, tau)
            )
            return updated_actor, updated_qf1, updated_qf2

        actor_state, qf1_state, qf2_state = jax.lax.cond(
            update_target_flag,
            update_target_networks,
            lambda c: c,
            (actor_state, qf1_state, qf2_state)
        )

        # temperature = jnp.clip(1.0 - 0.8 * global_step / config["TOTAL_TIMESTEPS"], 0.2, 1.0,)

        return (actor_state, qf1_state, qf2_state, buffer_state, next_env_state, next_obs, rng, global_step, temperature), (infos, qf1_loss, qf2_loss, actor_loss, qf1_val)

    def save_and_eval(step_count):
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        [
                            config,
                            td3_carry[0].params,
                            td3_carry[1].params,
                            td3_carry[2].params
                         ]
                    )
                )
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")

        eval_mods = config["EVAL_MODS"] if len(config["EVAL_MODS"]) > 0 else config["TRAIN_MODS"]
        eval_configs = [([], "default")]
        if len(eval_mods) > 0:
            mods_list = list(eval_mods)
            for mod in mods_list:
                mods_config = [mod] if not isinstance(mod, (list, tuple)) else list(mod)
                mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_config)
                eval_configs.append((mods_config, mod_label))

        metrics = {}
        for mods_cfg, mod_label in eval_configs:
            episodic_returns, env_states = evaluate(
                model_path,
                partial(
                    make_env,
                    mods=mods_cfg,
                    pixel_based=config["PIXEL_BASED"],
                    native_downscaling=config["NATIVE_DOWNSCALING"],
                    eval=True,
                ),
                config["ENV_ID"],
                eval_episodes=10,
                Model=(Pixel_Actor_Discrete, Pixel_Critic) if config["PIXEL_BASED"] else (MLP_Actor_Discrete, MLP_Critic),
                seed=config["SEED"]+42,
            )
            metrics[mod_label] = np.mean(jax.device_get(episodic_returns))
            wandb.log({f"eval/episodic_return_{mod_label}": np.mean(jax.device_get(episodic_returns))}, step=step_count)

            if config["CAPTURE_VIDEO"]: 
                clean_renderer = jaxatari.make(config["ENV_ID"], mods=mods_cfg).renderer
                frames = jax.vmap(clean_renderer.render)(env_states)
                frames = jnp.transpose(frames, (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                wandb.log({f"eval/video_{mod_label}": video}, step=step_count)
                print(f"Video (eval) logged to wandb with {frames.shape[0]} frames ({mod_label}).")
        return metrics

    print(f"[td3] start compile...")
    start_compile = time.perf_counter()
    global_step = jnp.array(0, dtype=jnp.int32)
    temperature = jnp.array(1.0)
    td3_carry = (actor_state, qf1_state, qf2_state, buffer_state, _state, _obs, key, global_step, temperature)

    @jax.jit
    def scanned_steps(carry):
        def step_fn(c, _):
            return full_td3_step(*c)
        return jax.lax.scan(step_fn, carry, None, length=config.get("SCAN_STEPS", 1000))

    _ = jax.block_until_ready(scanned_steps(td3_carry))
    end_compile = time.perf_counter()
    print(f"[td3] compilation time: {end_compile - start_compile:.2f}s")
    
    steps_per_iteration = config.get("NUM_ENVS") * config.get("TRAIN_FREQUENCY") * config.get("SCAN_STEPS")
    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=config.get("TOTAL_TIMESTEPS") // steps_per_iteration)
    rtpt.start()
    run_time = time.perf_counter()
    
    print(f"[td3] starting training for {config.get('TOTAL_TIMESTEPS')} steps...")
    while global_step < config.get("TOTAL_TIMESTEPS"):
        rtpt.step()
        iteration = global_step // steps_per_iteration
        if config["EVAL_DURING_TRAIN"] and iteration > 0 and iteration % config["EVAL_EVERY"] == 0:
           save_and_eval(global_step) 
           
        iteration_time_start = time.perf_counter()
        result = scanned_steps(td3_carry)
        td3_carry, (infos, qf1_loss, qf2_loss, actor_loss, qf1_val) = result
        global_step = int(td3_carry[-2])
        
        print(f"[td3] iteration {iteration} | step {global_step} | avg_return {infos['returned_episode_returns'][-1].mean():.2f} | qf1_loss {qf1_loss[-1]:.4f} | act_loss {actor_loss[-1]:.4f} | SPS {int(global_step / (time.perf_counter() - run_time))}")
        
        metrics = {
            "charts/avg_episodic_return": infos["returned_episode_returns"][-1].mean(), 
            "charts/avg_episodic_length": infos["returned_episode_lengths"][-1].mean(),
            "losses/qf1_loss": qf1_loss[-1].item(),
            "losses/qf2_loss": qf2_loss[-1].item(),
            "losses/actor_loss": actor_loss[-1].item(),
            "losses/qf1_values": qf1_val[-1].item(), 
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(config["NUM_ENVS"] * config["TRAIN_FREQUENCY"] * config["SCAN_STEPS"]  / (time.perf_counter() - iteration_time_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        wandb.log(metrics, step=global_step)

    eval_metrics = save_and_eval(global_step+1)
    wandb.finish()
    return eval_metrics