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
from flax.training.train_state import TrainState
import jaxatari
from jaxatari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    LogWrapper,
    FlattenObservationWrapper,
)


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


class NoisyDense(nn.Module):
    features: int
    sigma0: float = 0.5

    @nn.compact
    def __call__(self, x, deterministic=False):
        in_features = x.shape[-1]
        bound = 1.0 / np.sqrt(in_features)
        sigma_init = self.sigma0 / np.sqrt(in_features)

        def mu_init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype, -bound, bound)

        kernel_mu = self.param("kernel_mu", mu_init, (in_features, self.features))
        kernel_sigma = self.param(
            "kernel_sigma", nn.initializers.constant(sigma_init), (in_features, self.features)
        )
        bias_mu = self.param("bias_mu", mu_init, (self.features,))
        bias_sigma = self.param(
            "bias_sigma", nn.initializers.constant(sigma_init), (self.features,)
        )

        if deterministic:
            return x @ kernel_mu + bias_mu

        key = self.make_rng("noise")
        k_in, k_out = jax.random.split(key)
        f = lambda v: jnp.sign(v) * jnp.sqrt(jnp.abs(v))
        eps_in = f(jax.random.normal(k_in, (in_features,)))
        eps_out = f(jax.random.normal(k_out, (self.features,)))
        kernel = kernel_mu + kernel_sigma * jnp.outer(eps_in, eps_out)
        bias = bias_mu + bias_sigma * eps_out
        return x @ kernel + bias


class RainbowCNNNetwork(nn.Module):
    action_dim: int
    n_atoms: int
    sigma0: float = 0.5

    @nn.compact
    def __call__(self, x, deterministic=False):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32) / 255.0
        x = nn.relu(nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x))
        x = x.reshape((x.shape[0], -1))

        v = nn.relu(NoisyDense(512, self.sigma0)(x, deterministic))
        v = NoisyDense(self.n_atoms, self.sigma0)(v, deterministic)

        a = nn.relu(NoisyDense(512, self.sigma0)(x, deterministic))
        a = NoisyDense(self.action_dim * self.n_atoms, self.sigma0)(a, deterministic)
        a = a.reshape((a.shape[0], self.action_dim, self.n_atoms))

        logits = v[:, None, :] + a - jnp.mean(a, axis=1, keepdims=True)
        return jax.nn.softmax(logits, axis=-1)


class RainbowMLPNetwork(nn.Module):
    action_dim: int
    n_atoms: int
    sigma0: float = 0.5

    @nn.compact
    def __call__(self, x, deterministic=False):
        x = x.astype(jnp.float32)
        x = nn.relu(NoisyDense(461, self.sigma0)(x, deterministic))

        v = nn.relu(NoisyDense(512, self.sigma0)(x, deterministic))
        v = NoisyDense(self.n_atoms, self.sigma0)(v, deterministic)

        a = nn.relu(NoisyDense(512, self.sigma0)(x, deterministic))
        a = NoisyDense(self.action_dim * self.n_atoms, self.sigma0)(a, deterministic)
        a = a.reshape((a.shape[0], self.action_dim, self.n_atoms))

        logits = v[:, None, :] + a - jnp.mean(a, axis=1, keepdims=True)
        return jax.nn.softmax(logits, axis=-1)


class RainbowTrainState(TrainState):
    target_params: flax.core.FrozenDict
    atoms: jnp.ndarray


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


def build_eval_fn(env, apply_fn, v_min, v_max, n_atoms, eval_episodes, max_steps, action_dim):
    atoms = jnp.linspace(v_min, v_max, n_atoms)

    def wrapped_reset(key):
        next_obs, state = env.reset(key)
        return next_obs.squeeze()[None, ...], state

    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    def get_action(params, obs, key, epsilon):
        pmfs = apply_fn(params, obs, True)
        q_values = (pmfs * atoms[None, None, :]).sum(-1)
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

    run_name = f"{config['ENV_ID']}_{config['EXP_NAME']}_{'oc' if not config['PIXEL_BASED'] else 'pixel'}_{config['SEED']}"

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
    )

    # try not to modify the seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    env = make_env(
        config.get("ENV_ID"),
        list(config.get("TRAIN_MODS", [])),
        config.get("PIXEL_BASED", True),
        config.get("NATIVE_DOWNSCALING", True),
        False,
    )()

    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape
    if config.get("PIXEL_BASED", True):
        obs_shape = obs_shape[:-1]

    num_envs = config["NUM_ENVS"]

    @jax.jit
    def vmap_reset(rng):
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(rng.shape[0], *obs_shape), state

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs.reshape(action.shape[0], *obs_shape), state, reward, next_done, info

    n_atoms = config.get("N_ATOMS", 51)
    v_min = config.get("V_MIN", -10.0)
    v_max = config.get("V_MAX", 10.0)
    atoms = jnp.linspace(v_min, v_max, n_atoms)
    delta_z = (v_max - v_min) / (n_atoms - 1)

    n_step = config.get("N_STEP", 3)
    gamma = config.get("GAMMA", 0.99)
    gamma_n = gamma ** n_step
    batch_size = config.get("BATCH_SIZE", 32)
    beta_start = config.get("IS_BETA_START", 0.4)
    beta_end = config.get("IS_BETA_END", 1.0)
    total_timesteps = config.get("TOTAL_TIMESTEPS", 10000000)
    sigma0 = config.get("NOISY_SIGMA0", 0.5)

    key, q_key, noise_key = jax.random.split(key, 3)
    if config.get("PIXEL_BASED", True):
        network = RainbowCNNNetwork(action_dim=action_dim, n_atoms=n_atoms, sigma0=sigma0)
    else:
        network = RainbowMLPNetwork(action_dim=action_dim, n_atoms=n_atoms, sigma0=sigma0)

    dummy_obs = jnp.zeros((1, *obs_shape))
    q_params = network.init({"params": q_key, "noise": noise_key}, dummy_obs, False)

    tx = optax.adam(
        learning_rate=config.get("LEARNING_RATE", 0.0000625),
        eps=config.get("ADAM_EPS", 0.00015),
    )

    agent_state = RainbowTrainState.create(
        apply_fn=network.apply,
        params=q_params,
        target_params=jax.tree.map(jnp.copy, q_params),
        atoms=atoms,
        tx=tx,
    )

    obs_dtype = jnp.uint8 if config.get("PIXEL_BASED", True) else jnp.float32
    replay_buffer = fbx.make_prioritised_item_buffer(
        max_length=config.get("BUFFER_SIZE", 50000),
        min_length=config.get("LEARNING_STARTS", 20000),
        sample_batch_size=batch_size,
        add_batches=True,
        priority_exponent=config.get("PRIORITY_EXPONENT", 0.5),
        device="gpu" if jax.default_backend() == "gpu" else "cpu",
    )
    example_transition = {
        "obs": jnp.zeros(obs_shape, dtype=obs_dtype),
        "action": jnp.zeros((), dtype=jnp.int32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "done": jnp.zeros((), dtype=jnp.bool_),
        "next_obs": jnp.zeros(obs_shape, dtype=obs_dtype),
    }
    buffer_state = replay_buffer.init(example_transition)

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
    )

    eval_mods_list = list(config.get("EVAL_MODS", [])) or list(config.get("TRAIN_MODS", []))
    eval_configs = [([], "default")]
    for mod in eval_mods_list:
        mods_cfg = list(mod) if isinstance(mod, (list, tuple)) else [mod]
        mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_cfg)
        eval_configs.append((mods_cfg, mod_label))

    eval_episodes = 10
    eval_max_steps = 10000

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
            v_min=v_min,
            v_max=v_max,
            n_atoms=n_atoms,
            eval_episodes=eval_episodes,
            max_steps=eval_max_steps,
            action_dim=action_dim,
        )

    def step_once(carry, unused_step):
        state, buffer_state, env_state, obs, window, rng, global_step, ep_stats = carry

        rng, act_noise_rng = jax.random.split(rng)
        pmfs = state.apply_fn(state.params, obs, False, rngs={"noise": act_noise_rng})
        q_values = (pmfs * state.atoms[None, None, :]).sum(-1)
        actions = q_values.argmax(axis=-1)

        next_obs, next_env_state, rewards, next_done, infos = vmap_step(env_state, actions)

        new_returns = ep_stats.episode_returns + rewards
        new_lengths = ep_stats.episode_lengths + 1
        ep_stats = ep_stats.replace(
            episode_returns=new_returns * (1 - next_done),
            episode_lengths=new_lengths * (1 - next_done),
            returned_episode_returns=jnp.where(next_done, new_returns, ep_stats.returned_episode_returns),
            returned_episode_lengths=jnp.where(next_done, new_lengths, ep_stats.returned_episode_lengths),
        )

        w_obs, w_act, w_rew, w_done = window
        w_obs = jnp.concatenate([w_obs[1:], obs.astype(obs_dtype)[None]], axis=0)
        w_act = jnp.concatenate([w_act[1:], actions.astype(jnp.int32)[None]], axis=0)
        w_rew = jnp.concatenate([w_rew[1:], rewards.astype(jnp.float32)[None]], axis=0)
        w_done = jnp.concatenate([w_done[1:], next_done.astype(jnp.float32)[None]], axis=0)
        window = (w_obs, w_act, w_rew, w_done)

        not_done = 1.0 - w_done
        live = jnp.cumprod(not_done, axis=0)
        reward_mask = jnp.concatenate([jnp.ones((1, num_envs)), live[:-1]], axis=0)
        discounts = (gamma ** jnp.arange(n_step, dtype=jnp.float32))[:, None]
        n_step_return = jnp.sum(w_rew * reward_mask * discounts, axis=0)
        n_step_done = 1.0 - live[-1]

        transition = {
            "obs": w_obs[0],
            "action": w_act[0],
            "reward": n_step_return,
            "done": n_step_done.astype(jnp.bool_),
            "next_obs": next_obs.astype(obs_dtype),
        }
        window_full = global_step >= (n_step - 1) * num_envs
        buffer_state = jax.lax.cond(
            window_full,
            lambda bs: replay_buffer.add(bs, transition),
            lambda bs: bs,
            buffer_state,
        )

        beta = jnp.interp(
            global_step,
            jnp.array([0, total_timesteps]),
            jnp.array([beta_start, beta_end]),
        )

        updates_per_step = max(1, num_envs // config.get("TRAIN_FREQUENCY", 4))

        def do_update(update_carry, _):
            u_state, u_buffer_state, u_key = update_carry
            u_key, sample_key, select_noise, target_noise, online_noise = jax.random.split(u_key, 5)

            sampled = replay_buffer.sample(u_buffer_state, sample_key)
            batch = sampled.experience
            b_obs = batch["obs"]
            b_act = batch["action"].reshape(-1)
            b_rew = batch["reward"]
            b_don = batch["done"].astype(jnp.float32)
            b_nobs = batch["next_obs"]

            is_weights = (1.0 / (sampled.probabilities + 1e-10)) ** beta
            is_weights = is_weights / jnp.max(is_weights)

            next_pmfs_online = u_state.apply_fn(
                u_state.params, b_nobs, False, rngs={"noise": select_noise}
            )
            next_q_online = (next_pmfs_online * u_state.atoms[None, None, :]).sum(-1)
            next_action = jnp.argmax(next_q_online, axis=-1)

            next_pmfs = u_state.apply_fn(
                u_state.target_params, b_nobs, False, rngs={"noise": target_noise}
            )
            next_pmfs = next_pmfs[jnp.arange(batch_size), next_action]

            next_atoms = b_rew[:, None] + gamma_n * u_state.atoms[None, :] * (1.0 - b_don[:, None])
            tz = jnp.clip(next_atoms, v_min, v_max)
            b = (tz - v_min) / delta_z
            l = jnp.clip(jnp.floor(b).astype(jnp.int32), 0, n_atoms - 1)
            u = jnp.clip(jnp.ceil(b).astype(jnp.int32), 0, n_atoms - 1)
            d_l = (u.astype(jnp.float32) + (l == u).astype(jnp.float32) - b) * next_pmfs
            d_u = (b - l.astype(jnp.float32)) * next_pmfs

            target_pmfs = jnp.zeros((batch_size, n_atoms))

            def project_sample(i, val):
                val = val.at[i, l[i]].add(d_l[i])
                val = val.at[i, u[i]].add(d_u[i])
                return val

            target_pmfs = jax.lax.fori_loop(0, batch_size, project_sample, target_pmfs)
            target_pmfs = jax.lax.stop_gradient(target_pmfs)

            def loss_fn(params):
                pmfs = u_state.apply_fn(params, b_obs, False, rngs={"noise": online_noise})
                p = pmfs[jnp.arange(batch_size), b_act]
                p = jnp.clip(p, 1e-5, 1.0)
                cross_entropy = -(target_pmfs * jnp.log(p)).sum(-1)
                loss = jnp.mean(is_weights * cross_entropy)
                return loss, cross_entropy

            (loss, cross_entropy), grads = jax.value_and_grad(loss_fn, has_aux=True)(u_state.params)
            new_state = u_state.apply_gradients(grads=grads)

            target_entropy_term = jnp.sum(
                target_pmfs * jnp.log(jnp.clip(target_pmfs, 1e-5, 1.0)), axis=-1
            )
            kl = jnp.clip(target_entropy_term + cross_entropy, 1e-6, None)
            new_buffer_state = replay_buffer.set_priorities(u_buffer_state, sampled.indices, kl)

            return (new_state, new_buffer_state, u_key), loss

        def run_updates(s_state, s_buffer, s_key):
            (new_state, new_buffer, new_key), losses = jax.lax.scan(
                do_update, (s_state, s_buffer, s_key), None, length=updates_per_step
            )
            return new_state, new_buffer, new_key, jnp.mean(losses)

        should_train_step = (global_step % config.get("TRAIN_FREQUENCY", 4)) < num_envs
        can_train = jnp.logical_and(replay_buffer.can_sample(buffer_state), should_train_step)

        state, buffer_state, rng, avg_loss = jax.lax.cond(
            can_train,
            lambda c: run_updates(c[0], c[1], c[2]),
            lambda c: (c[0], c[1], c[2], 0.0),
            (state, buffer_state, rng),
        )

        update_target_flag = jnp.logical_and(
            can_train,
            (global_step % config.get("TARGET_NETWORK_FREQUENCY", 8000)) < num_envs,
        )
        new_target_params = jax.lax.cond(
            update_target_flag,
            lambda _: optax.incremental_update(state.params, state.target_params, config.get("TAU", 1.0)),
            lambda _: state.target_params,
            None,
        )
        state = state.replace(target_params=new_target_params)

        global_step += num_envs
        return (state, buffer_state, next_env_state, next_obs, window, rng, global_step, ep_stats), (avg_loss, beta)

    def save_and_eval(step_count, agent_state):
        model_path = ""
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes((None, agent_state.params)))
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")

        metrics = {}
        for mods_cfg, mod_label in eval_configs:
            reset_keys = jax.random.split(jax.random.PRNGKey(config["SEED"]), eval_episodes)

            episodic_returns, first_states_history, first_done = eval_fns[mod_label](
                agent_state.params, reset_keys, 0.0
            )

            avg_eval_return = float(jnp.mean(episodic_returns))
            return_key = f"eval/episodic_return_{mod_label}"
            metrics[return_key] = avg_eval_return
            print(f"evaluation at step {step_count} ({mod_label}): average return = {avg_eval_return}")

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

    CHUNK_SIZE = config["NUM_STEPS"] // num_envs

    @partial(jax.jit, donate_argnums=(0,))
    def rollout_chunk(carry):
        return jax.lax.scan(step_once, carry, None, length=CHUNK_SIZE)

    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, num_envs))
    global_step = jnp.array(0, dtype=jnp.int32)

    window = (
        jnp.zeros((n_step, num_envs, *obs_shape), dtype=obs_dtype),
        jnp.zeros((n_step, num_envs), dtype=jnp.int32),
        jnp.zeros((n_step, num_envs), dtype=jnp.float32),
        jnp.zeros((n_step, num_envs), dtype=jnp.float32),
    )

    carry = (agent_state, buffer_state, env_state, obs, window, key, global_step, episode_stats)

    start_time = time.time()
    total_eval_time = 0.0
    total_iterations = total_timesteps // (num_envs * CHUNK_SIZE)

    print(f"starting compilation and run ({total_iterations} chunks of {CHUNK_SIZE * num_envs} steps)")

    for i in range(1, total_iterations + 1):
        iteration_time_start = time.time()
        carry, (losses, betas) = rollout_chunk(carry)

        agent_state, buffer_state, env_state, obs, window, key, global_step, episode_stats = carry

        current_step = global_step.item()
        iteration_time = time.time() - iteration_time_start

        if config.get("EVAL_DURING_TRAIN", True) and (i % config.get("EVAL_EVERY", 10) == 0):
            eval_t0 = time.time()
            save_and_eval(current_step, agent_state)
            total_eval_time += time.time() - eval_t0

        metrics = {
            "charts/avg_episodic_return": episode_stats.returned_episode_returns.mean().item(),
            "charts/avg_episodic_length": episode_stats.returned_episode_lengths.mean().item(),
            "charts/is_beta": betas[-1].item(),
            "charts/SPS": int(current_step / (time.time() - start_time - total_eval_time)),
            "charts/SPS_update": int(CHUNK_SIZE * num_envs / iteration_time),
            "losses/td_loss": float(jnp.sum(losses) / jnp.maximum(jnp.sum(losses != 0), 1)),
            "charts/global_step": current_step,
        }
        wandb.log(metrics, step=current_step)

        if i % (max(1, total_iterations // 20)) == 0:
            sps = int(current_step / (time.time() - start_time - total_eval_time))
            print(f"step: {current_step} / {total_timesteps} | SPS: {sps} | return: {episode_stats.returned_episode_returns.mean().item():.2f}")

    eval_metrics = save_and_eval(total_timesteps, agent_state)

    wandb.finish()

    return eval_metrics
