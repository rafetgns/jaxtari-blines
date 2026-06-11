# Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari_jax.py
import os
import random
import time
from functools import partial
from typing import Sequence, NamedTuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import jaxatari
from jaxatari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    LogWrapper,
    FlattenObservationWrapper
)
from jaxatari import spaces


def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False):
    def thunk():


        active_mods = mods
        if not eval and isinstance(active_mods, (list, tuple)) and len(active_mods) > 1:
            active_mods = []                                                             


       
        if isinstance(active_mods, (list, tuple)) and len(active_mods) == 0:
            mods_arg = None
        else:
            mods_arg = active_mods

        env = jaxatari.make(env_id, mods=mods_arg)



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


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32)
        x = x / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class MLP_QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(461, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x


class DQNTrainState(TrainState):
    target_params: flax.core.FrozenDict


@flax.struct.dataclass 
class ReplayBuffer:
    obs: jnp.array
    actions: jnp.array
    rewards: jnp.array
    dones: jnp.array
    next_obs: jnp.array
    pos: jnp.array
    full: jnp.array

    @classmethod
    def create(cls, capacity, num_envs, obs_shape, obs_dtype=jnp.uint8):
        """pre-allocates the buffer space via GPU."""
        return cls(
            obs=jnp.zeros((capacity, num_envs, *obs_shape), dtype=obs_dtype),
            actions=jnp.zeros((capacity, num_envs), dtype=jnp.int32),
            rewards=jnp.zeros((capacity, num_envs), dtype=jnp.float32),
            dones=jnp.zeros((capacity, num_envs), dtype=jnp.bool_),
            next_obs=jnp.zeros((capacity, num_envs, *obs_shape), dtype=obs_dtype),
            pos=jnp.array(0, dtype=jnp.int32),
            full=jnp.array(False, dtype=jnp.bool_),
        )

   
    def add(self, obs, action, reward, done, next_obs):
        new_obs = self.obs.at[self.pos].set(obs)
        new_actions = self.actions.at[self.pos].set(action)
        new_rewards = self.rewards.at[self.pos].set(reward)
        new_dones = self.dones.at[self.pos].set(done)
        new_next_obs = self.next_obs.at[self.pos].set(next_obs)

        capacity = self.obs.shape[0]
        new_pos = self.pos + 1
        new_full = jnp.logical_or(self.full, new_pos >= capacity)
        new_pos = new_pos % capacity

        return self.replace(
            obs=new_obs, actions=new_actions, rewards=new_rewards,
            dones=new_dones, next_obs=new_next_obs,
            pos=new_pos, full=new_full
        )

    def sample(self, key, batch_size):
        capacity = self.obs.shape[0]
        num_envs = self.obs.shape[1]
        max_idx = jnp.where(self.full, capacity, self.pos) 

        key1, key2 = jax.random.split(key)
        
        idx = jax.random.randint(key1, (batch_size,), 0, max_idx)
        env_idx = jax.random.randint(key2, (batch_size,), 0, num_envs)

        return (
            self.obs[idx, env_idx],
            self.actions[idx, env_idx],
            self.rewards[idx, env_idx],
            self.dones[idx, env_idx],
            self.next_obs[idx, env_idx]
        )

@flax.struct.dataclass
class EpisodeStatistics: 
    
    episode_returns: jnp.array
    episode_lengths: jnp.array

    
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array

    
    





def build_eval_fn(env, apply_fn, eval_episodes, max_steps, action_dim):
   

    def wrapped_reset(key):
        next_obs, state = env.reset(key)

        return next_obs.squeeze()[None, ...], state

    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    def get_action(params, obs, key, epsilon):
        q_values = apply_fn(params, obs)
        greedy_action = jnp.argmax(q_values, axis=1)

        key, subkey = jax.random.split(key)
        random_action = jax.random.randint(subkey, greedy_action.shape, 0, action_dim)
        explore = jax.random.uniform(key, greedy_action.shape) < epsilon
        action = jnp.where(explore, random_action, greedy_action)
        return action, key

    def step_fn(carry, _):
        obs, env_state, keys, params, epsilon = carry

        actions, keys = jax.vmap(get_action, in_axes=(None, 0, 0, None))(params, obs, keys, epsilon)
        next_obs, next_env_state, reward, done, info = jax.vmap(wrapped_step)(env_state, actions)

        
        first_state = jax.tree.map(lambda x: x[0], next_env_state)

        return (next_obs, next_env_state, keys, params, epsilon), (first_state, done, reward)

    @jax.jit
    def eval_fn(params, reset_keys, epsilon):
        obs, env_state = jax.vmap(wrapped_reset)(reset_keys)

        _, (first_states_history, dones, rewards) = jax.lax.scan(
            step_fn, (obs, env_state, reset_keys, params, epsilon), None, length=max_steps
        )

       
        has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
        mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
        masked_rewards = rewards * (1 - mask_after_first_done)
        episodic_returns = jnp.sum(masked_rewards, axis=0)

        first_done = jnp.argmax(dones, axis=0)
        return episodic_returns, first_states_history, first_done

    return eval_fn



def single_run(config: dict):
    
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])
    if isinstance(config.get("EVAL_MODS"), list):
        config["EVAL_MODS"] = tuple(config["EVAL_MODS"])

   
    if config.get("PIXEL_BASED", True) and config.get("NUM_ENVS", 1) > 16:
        config["NUM_ENVS"] = 8

    
    run_name = f"{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}_{config["SEED"]}"

    if config.get("TRACK", True): 
        wandb.init(
            project=config.get("PROJECT", "jaxtari-blines"),
            entity=config.get("ENTITY", None),
            config=config,
            name=run_name,
            save_code=True,
        )

    # do not modify the seeding 
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    
    env = make_env(
        config.get("ENV_ID"),
        list(config.get("TRAIN_MODS", [])),
        config.get("PIXEL_BASED", True),
        config.get("NATIVE_DOWNSCALING", True),
        False
    )()

    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape
    if config.get("PIXEL_BASED", True):
        obs_shape = obs_shape[:-1]  

    @jax.jit 
    def vmap_reset(rng): 
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(rng.shape[0], *obs_shape), state
   

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs.reshape(action.shape[0], *obs_shape), state, reward, next_done, info

   
    capacity_per_env = config.get("BUFFER_SIZE", 1000000) // config.get("NUM_ENVS", 8)

    
    key, q_key = jax.random.split(key, 2) 
    network = QNetwork(action_dim=action_dim) if config.get("PIXEL_BASED", True) else MLP_QNetwork(action_dim=action_dim)

    dummy_obs = jnp.zeros((1, *obs_shape)) 
    q_params = network.init(q_key, dummy_obs)
    




    tx = optax.adam(learning_rate=config.get("LEARNING_RATE"), eps=1e-4)

    agent_state = DQNTrainState.create(
        apply_fn=network.apply,
        params=q_params,
        target_params=jax.tree.map(jnp.copy, q_params),
        tx=tx,
    )
    
    obs_dtype = jnp.uint8 if config.get("PIXEL_BASED", True) else jnp.float32
    replay_buffer = ReplayBuffer.create(capacity_per_env, config.get("NUM_ENVS", 8), obs_shape, obs_dtype)

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(config["NUM_ENVS"], dtype=jnp.float32),
        episode_lengths=jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(config["NUM_ENVS"], dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32),
    )

    eval_mods_list = list(config.get("EVAL_MODS", [])) or list(config.get("TRAIN_MODS", []))
    eval_configs = [([], "default")]
    for mod in eval_mods_list:
        mods_cfg = list(mod) if isinstance(mod, (list, tuple)) else [mod]
        mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_cfg)
        eval_configs.append((mods_cfg, mod_label))

    eval_episodes = config.get("EVAL_EPISODES", 10)
    eval_max_steps = config.get("EVAL_MAX_STEPS", 3000)

    eval_fns = {}
    for mods_cfg, mod_label in eval_configs:
        eval_env = make_env(
            config["ENV_ID"],
            mods=mods_cfg,
            pixel_based=config.get("PIXEL_BASED", True),
            native_downscaling=config.get("NATIVE_DOWNSCALING", True),
            eval=True,
        )()
        eval_fns[mod_label] = build_eval_fn(
            env=eval_env,
            apply_fn=network.apply,
            eval_episodes=eval_episodes,
            max_steps=eval_max_steps,
            action_dim=action_dim,
        )


    def step_once(carry, unused_step):
 

        state, buffer, env_state, obs, rng, global_step, ep_stats = carry


        rng, action_rng, explore_rng = jax.random.split(rng, 3)
        epsilon = jnp.interp(
            global_step,
            jnp.array([0, config.get("EXPLORATION_FRACTION", 0.10) * config.get("TOTAL_TIMESTEPS", 10000000)]),
            jnp.array([config.get("START_E", 1.0), config.get("END_E", 0.05)])
        )


        q_values = state.apply_fn(state.params, obs)
        greedy_actions = q_values.argmax(axis=-1)
        random_actions = jax.random.randint(action_rng, (config["NUM_ENVS"],), 0, action_dim)

        explore_mask = jax.random.uniform(explore_rng, (config["NUM_ENVS"],)) < epsilon
        actions = jnp.where(explore_mask, random_actions, greedy_actions)

        
        next_obs, next_env_state, rewards, next_done, infos = vmap_step(env_state, actions)

        
        buffer = buffer.add(obs, actions, rewards, next_done, next_obs)

        
        new_returns = ep_stats.episode_returns + rewards
        new_lengths = ep_stats.episode_lengths + 1

        ep_stats = ep_stats.replace(
            episode_returns=new_returns * (1 - next_done),
            episode_lengths=new_lengths * (1 - next_done),
            returned_episode_returns=jnp.where(next_done, new_returns, ep_stats.returned_episode_returns),
            returned_episode_lengths=jnp.where(next_done, new_lengths, ep_stats.returned_episode_lengths),
        )


        updates_per_step = max(1, config["NUM_ENVS"] // config.get("TRAIN_FREQUENCY", 4))

        def do_update(update_carry, _):
            u_state, u_key = update_carry
            u_key, sample_key = jax.random.split(u_key)

            b_obs, b_act, b_rew, b_don, b_nobs = buffer.sample(sample_key, config.get("BATCH_SIZE", 32))

            def q_loss_fn(params):
                q_pred = u_state.apply_fn(params, b_obs) #forward feed
                q_pred = q_pred[jnp.arange(config.get("BATCH_SIZE", 32)), b_act.reshape(-1)]

                q_next = u_state.apply_fn(u_state.target_params, b_nobs)
                target = jax.lax.stop_gradient(
                    b_rew + (1.0 - b_don) * config.get("GAMMA", 0.99) * q_next.max(axis=-1)
                )

                error = q_pred - target

                if config.get("USE_HUBER_LOSS"):
                    loss = jnp.mean(optax.huber_loss(error))
                else:
                    loss = jnp.mean(error ** 2)

                return loss, q_pred

            (loss, q_val), grads = jax.value_and_grad(q_loss_fn, has_aux=True)(u_state.params)
            new_state = u_state.apply_gradients(grads=grads)

            return (new_state, u_key), loss

        def run_updates(s_state, s_key):
            (new_s_state, new_s_key), losses = jax.lax.scan(do_update, (s_state, s_key), None, length=updates_per_step)
            return new_s_state, new_s_key, jnp.mean(losses)


        should_train_step = (global_step % config.get("TRAIN_FREQUENCY", 4)) < config["NUM_ENVS"]
        can_train = jnp.logical_and(global_step > config.get("LEARNING_STARTS", 80000), should_train_step)

        state, rng, avg_loss = jax.lax.cond(
            can_train,
            lambda c: run_updates(c[0], c[1]),
            lambda c: (c[0], c[1], 0.0),
            (state, rng)
        )


        update_target_flag = jnp.logical_and(
            can_train,
            (global_step % config.get("TARGET_NETWORK_FREQUENCY", 1000)) < config["NUM_ENVS"]
        )

        new_target_params = jax.lax.cond(
            update_target_flag,
            lambda _: optax.incremental_update(state.params, state.target_params, config.get("TAU", 1.0)),
            lambda _: state.target_params,
            None
        )
        state = state.replace(target_params=new_target_params)

        global_step += config["NUM_ENVS"]
        return (state, buffer, next_env_state, next_obs, rng, global_step, ep_stats), (avg_loss, epsilon)



    def save_and_eval(step_count, agent_state):
        model_path = ""
        if config.get("SAVE_PATH", "./models") is not None:

            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)


            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        (None, agent_state.params)
                    )
                )
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")

        metrics = {}
        for mods_cfg, mod_label in eval_configs:

            reset_keys = jax.random.split(jax.random.PRNGKey(config["SEED"]), eval_episodes)

            episodic_returns, first_states_history, first_done = eval_fns[mod_label](
                agent_state.params, reset_keys, 0.05
            )

            avg_eval_return = float(jnp.mean(episodic_returns))
            return_key = f"eval/episodic_return_{mod_label}" 
            metrics[return_key] = avg_eval_return
            print(f"evaluation at step {step_count} ({mod_label}): average return = {avg_eval_return}")

            if config.get("TRACK", True):
                wandb.log({return_key: avg_eval_return}, step=step_count)

                if config.get("CAPTURE_VIDEO", False):
                    clean_renderer = jaxatari.make(config["ENV_ID"], mods=mods_cfg).renderer
                    env_states_until_done = jax.tree.map(
                        lambda x: x[: first_done[0] + 1],
                        first_states_history.atari_state.atari_state.env_state,
                    )
                    frames = jax.vmap(clean_renderer.render)(env_states_until_done)
                    frames = jnp.transpose(frames, (0, 3, 1, 2))
                    video = wandb.Video(np.array(frames), fps=30, format="mp4")

                    video_key = f"eval/video_{mod_label}"
                    wandb.log({video_key: video}, step=step_count)
                    print(f"video (eval) logged to wandb with {frames.shape} frames ({mod_label}).")

        return metrics


    CHUNK_SIZE = 2_000 // config["NUM_ENVS"]

    @partial(jax.jit, donate_argnums=(0,))
    def rollout_chunk(carry):
        return jax.lax.scan(step_once, carry, None, length=CHUNK_SIZE)



    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, config["NUM_ENVS"]))
    global_step = jnp.array(0, dtype=jnp.int32)

    carry = (agent_state, replay_buffer, env_state, obs, key, global_step, episode_stats)

    start_time = time.time()
    total_eval_time = 0.0
    total_iterations = config.get("TOTAL_TIMESTEPS", 10000000) // (config["NUM_ENVS"] * CHUNK_SIZE)

    print(f"starting compilation and run ({total_iterations} chunks of {CHUNK_SIZE * config['NUM_ENVS']} steps)")


    for i in range(1, total_iterations + 1):
        carry, (losses, epsilons) = rollout_chunk(carry)


        agent_state, replay_buffer, env_state, obs, key, global_step, episode_stats = carry

        current_step = global_step.item()


        if config.get("EVAL_DURING_TRAIN", True) and (i % config.get("EVAL_EVERY", 10) == 0):
            eval_t0 = time.time()
            save_and_eval(current_step, agent_state)
            total_eval_time += time.time() - eval_t0

        if config.get("TRACK", True):

            metrics = {
                "charts/avg_episodic_return": episode_stats.returned_episode_returns.mean().item(),
                "charts/avg_episodic_length": episode_stats.returned_episode_lengths.mean().item(),
                "charts/epsilon": epsilons[-1].item(),
                "charts/SPS": int(current_step / (time.time() - start_time - total_eval_time)),
                "losses/td_loss": float(jnp.sum(losses) / jnp.maximum(jnp.sum(losses != 0), 1)),
                "charts/global_step": current_step,
            }
            wandb.log(metrics, step=current_step)

        if i % (max(1, total_iterations // 20)) == 0:
            sps = int(current_step / (time.time() - start_time - total_eval_time))
            print(f"step: {current_step} / {config.get('TOTAL_TIMESTEPS')} | SPS: {sps} | return: {episode_stats.returned_episode_returns.mean().item():.2f}")


    eval_metrics = save_and_eval(config.get("TOTAL_TIMESTEPS", 10000000), agent_state)

    if config.get("TRACK", True):
        wandb.finish()

    return eval_metrics


