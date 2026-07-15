import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flashbax as fbx
import wandb
import flax
import random

import jaxatari
from agents.rainbow.rainbow import (
    make_env,
    RainbowCNNNetwork,
    RainbowMLPNetwork,
    RainbowTrainState,
    EpisodeStatistics,
    build_eval_fn,
)

try:
    import tqdx as _tqdx
except ImportError:
    _tqdx = None


def build_eval_return_fn(env, apply_fn, v_min, v_max, n_atoms, max_steps):
    atoms = jnp.linspace(v_min, v_max, n_atoms)

    def wrapped_reset(key):
        obs, state = env.reset(key)
        return obs.squeeze()[None, ...], state

    def wrapped_step(state, action):
        obs, state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return obs.squeeze()[None, ...], state, reward, done

    def get_action(params, obs):
        pmfs = apply_fn(params, obs, True)
        q_values = (pmfs * atoms[None, None, :]).sum(-1)
        return jnp.argmax(q_values, axis=1)

    def step_fn(carry, _):
        obs, state, params = carry
        actions = jax.vmap(get_action, in_axes=(None, 0))(params, obs)
        obs, state, reward, done = jax.vmap(wrapped_step)(state, actions)
        return (obs, state, params), (done, reward)

    def eval_return_fn(params, reset_keys):
        obs, state = jax.vmap(wrapped_reset)(reset_keys)
        _, (dones, rewards) = jax.lax.scan(
            step_fn, (obs, state, params), None, length=max_steps
        )
        has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
        mask = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
        masked = rewards * (1 - mask)
        return jnp.mean(jnp.sum(masked, axis=0))

    return eval_return_fn


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    if _tqdx is None:
        raise ImportError(
            "RAINBOW_SCAN needs tqdx: uv add 'tqdx @ git+https://github.com/huterguier/tqdx'"
        )

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])
    if isinstance(config.get("EVAL_MODS"), list):
        config["EVAL_MODS"] = tuple(config["EVAL_MODS"])

    def _as_mods(spec):
        if spec in (None, "default") or spec == [] or spec == ():
            return []
        return list(spec) if isinstance(spec, (list, tuple)) else [spec]

    def _label(mods_cfg):
        return "default" if not mods_cfg else "_".join(str(m) for m in mods_cfg)

    train_mods = _as_mods(config.get("TRAIN_ENV", "default"))
    if not train_mods and config.get("TRAIN_MODS"):
        train_mods = list(config["TRAIN_MODS"])
    train_label = _label(train_mods)

    eval_env_specs = config.get("EVAL_ENVS", None)
    if eval_env_specs is None:
        eval_env_specs = list(config.get("EVAL_MODS", [])) or list(config.get("TRAIN_MODS", []))

    eval_configs = [([], "default")]
    seen_labels = {"default"}
    for spec in eval_env_specs:
        mods_cfg = _as_mods(spec)
        mod_label = _label(mods_cfg)
        if mod_label not in seen_labels:
            seen_labels.add(mod_label)
            eval_configs.append((mods_cfg, mod_label))

    if config.get("PIXEL_BASED", True) and config.get("NUM_ENVS", 1) > 16:
        config["NUM_ENVS"] = 8

    env_tag = "" if train_label == "default" else f"_{train_label}"
    run_name = f"{config['ENV_ID']}_{config['EXP_NAME']}{env_tag}_{'oc' if not config['PIXEL_BASED'] else 'pixel'}_{config['SEED']}"

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
        max_length=config.get("BUFFER_SIZE", 1000000),
        min_length=config.get("LEARNING_STARTS", 80000),
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

    inscan_eval_env = make_env(
        config["ENV_ID"],
        mods=train_mods,
        pixel_based=config.get("PIXEL_BASED", True),
        native_downscaling=config.get("NATIVE_DOWNSCALING", True),
        eval=True,
    )()
    inscan_eval_fn = build_eval_return_fn(
        inscan_eval_env, network.apply, v_min, v_max, n_atoms, eval_max_steps
    )
    eval_reset_keys = jax.random.split(jax.random.PRNGKey(config["SEED"]), eval_episodes)

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
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes((None, agent_state.params)))
            print(f"model saved to {model_path}")

        metrics = {}
        for mods_cfg, mod_label in eval_configs:
            reset_keys = jax.random.split(jax.random.PRNGKey(config["SEED"]), eval_episodes)
            episodic_returns, first_states_history, first_done = eval_fns[mod_label](
                agent_state.params, reset_keys, 0.0
            )
            avg_eval_return = float(jnp.mean(episodic_returns))
            return_key = f"eval/episodic_return_{mod_label}"
            metrics[return_key] = avg_eval_return
            print(f"final eval ({mod_label}): average return = {avg_eval_return}")
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
                wandb.log({f"eval/video_{mod_label}": video}, step=step_count)
                print(f"video (eval) logged with {frames.shape} frames ({mod_label}).")
        return metrics

    CHUNK_SIZE = config["NUM_STEPS"] // num_envs
    total_iterations = total_timesteps // (num_envs * CHUNK_SIZE)
    eval_every = config.get("EVAL_EVERY", 100)
    eval_during_train = config.get("EVAL_DURING_TRAIN", True)

    steps_per_chunk = num_envs * CHUNK_SIZE
    _timing = {"start": None, "start_step": 0, "last": None}

    def log_cb(m):
        now = time.time()
        step = int(m["charts/global_step"])
        d = {k: float(v) for k, v in m.items()}
        d["charts/global_step"] = step
        if _timing["start"] is None:
            _timing.update(start=now, start_step=step, last=now)
        else:
            dt = now - _timing["last"]
            elapsed = now - _timing["start"]
            d["charts/SPS_update"] = int(steps_per_chunk / dt) if dt > 0 else 0
            d["charts/SPS"] = int((step - _timing["start_step"]) / elapsed) if elapsed > 0 else 0
            _timing["last"] = now
        wandb.log(d, step=step)

    def outer_step(carry, i):
        base_carry, last_eval = carry
        base_carry, (losses, betas) = jax.lax.scan(step_once, base_carry, None, length=CHUNK_SIZE)
        state, buffer_state, env_state, obs, window, rng, global_step, ep_stats = base_carry

        if eval_during_train:
            eval_return = jax.lax.cond(
                (i % eval_every) == 0,
                lambda p: inscan_eval_fn(p, eval_reset_keys),
                lambda p: last_eval,
                state.params,
            )
        else:
            eval_return = last_eval

        avg_loss = jnp.sum(losses) / jnp.maximum(jnp.sum(losses != 0), 1)
        metrics = {
            "charts/global_step": global_step,
            "charts/avg_episodic_return": ep_stats.returned_episode_returns.mean(),
            "charts/avg_episodic_length": ep_stats.returned_episode_lengths.mean().astype(jnp.float32),
            "charts/is_beta": betas[-1],
            "losses/td_loss": avg_loss,
        }
        if eval_during_train:
            metrics[f"eval/episodic_return_{train_label}"] = eval_return
        jax.debug.callback(log_cb, metrics)

        return (base_carry, eval_return), None

    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, num_envs))
    global_step = jnp.array(0, dtype=jnp.int32)

    window = (
        jnp.zeros((n_step, num_envs, *obs_shape), dtype=obs_dtype),
        jnp.zeros((n_step, num_envs), dtype=jnp.int32),
        jnp.zeros((n_step, num_envs), dtype=jnp.float32),
        jnp.zeros((n_step, num_envs), dtype=jnp.float32),
    )

    base_carry = (agent_state, buffer_state, env_state, obs, window, key, global_step, episode_stats)

    @partial(jax.jit, donate_argnums=(0,))
    def train(base_carry):
        if eval_during_train:
            init_eval = inscan_eval_fn(base_carry[0].params, eval_reset_keys)
        else:
            init_eval = jnp.float32(0.0)
        carry, _ = _tqdx.scan(outer_step, (base_carry, init_eval), jnp.arange(1, total_iterations + 1))
        return carry

    print(f"[rainbow_scan] compiling one scan of {total_iterations} chunks x {CHUNK_SIZE * num_envs} steps...")
    start_time = time.time()
    (base_carry, _last_eval) = jax.block_until_ready(train(base_carry))
    wall = time.time() - start_time

    agent_state = base_carry[0]
    total_steps = int(base_carry[6])
    print(f"[rainbow_scan] {total_steps} steps in {wall:.1f}s incl. compile -> {int(total_steps / wall)} SPS (compile-inclusive)")

    eval_metrics = save_and_eval(total_steps, agent_state)
    wandb.finish()
    return eval_metrics
