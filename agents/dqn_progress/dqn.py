import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flashbax as fbx
import wandb
import random

from agents.dqn.dqn import (
    make_env,
    QNetwork,
    MLP_QNetwork,
    DQNTrainState,
    EpisodeStatistics,
    build_eval_fn,
)


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    progress_mode = config.get("PROGRESS_MODE", "tqdm_outer")
    assert progress_mode in ("tqdm_outer", "tqdx_inner", "scan_outer"), progress_mode
    if progress_mode in ("tqdx_inner", "scan_outer") and _tqdx is None:
        raise ImportError(
            f"PROGRESS_MODE={progress_mode} needs tqdx: "
            "uv add 'tqdx @ git+https://github.com/huterguier/tqdx'"
        )
    if progress_mode == "tqdm_outer" and _tqdm is None:
        raise ImportError("PROGRESS_MODE=tqdm_outer needs tqdm (uv add tqdm)")

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
    wandb.define_metric("*", step_metric="charts/global_step")

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
    replay_buffer = fbx.make_item_buffer(
        max_length=config.get("BUFFER_SIZE", 1000000),
        min_length=config.get("LEARNING_STARTS", 80000),
        sample_batch_size=config.get("BATCH_SIZE", 32),
        add_batches=True,
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
            eval_episodes=eval_episodes,
            max_steps=eval_max_steps,
            action_dim=action_dim,
        )

    def step_once(carry, unused_step):
        state, buffer_state, env_state, obs, rng, global_step, ep_stats = carry

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

        transition = {
            "obs": obs.astype(obs_dtype),
            "action": actions.astype(jnp.int32),
            "reward": rewards.astype(jnp.float32),
            "done": next_done.astype(jnp.bool_),
            "next_obs": next_obs.astype(obs_dtype),
        }
        buffer_state = replay_buffer.add(buffer_state, transition)

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
            batch = replay_buffer.sample(buffer_state, sample_key).experience
            b_obs = batch["obs"]
            b_act = batch["action"]
            b_rew = batch["reward"]
            b_don = batch["done"]
            b_nobs = batch["next_obs"]

            def q_loss_fn(params):
                q_pred = u_state.apply_fn(params, b_obs)
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
        can_train = jnp.logical_and(replay_buffer.can_sample(buffer_state), should_train_step)

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
        return (state, buffer_state, next_env_state, next_obs, rng, global_step, ep_stats), (avg_loss, epsilon)

    def save_and_eval(step_count, agent_state):
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(flax_to_bytes(agent_state.params))
            print(f"model saved to {model_path}")
        metrics = {}
        for mods_cfg, mod_label in eval_configs:
            reset_keys = jax.random.split(jax.random.PRNGKey(config["SEED"]), eval_episodes)
            episodic_returns, _, _ = eval_fns[mod_label](agent_state.params, reset_keys, 0.05)
            avg_eval_return = float(jnp.mean(episodic_returns))
            return_key = f"eval/episodic_return_{mod_label}"
            metrics[return_key] = avg_eval_return
            print(f"evaluation at step {step_count} ({mod_label}): average return = {avg_eval_return}")
            wandb.log({return_key: avg_eval_return}, step=step_count)
        return metrics

    import flax
    def flax_to_bytes(params):
        return flax.serialization.to_bytes((None, params))

    CHUNK_SIZE = config["NUM_STEPS"] // config["NUM_ENVS"]
    total_iterations = config.get("TOTAL_TIMESTEPS", 10000000) // (config["NUM_ENVS"] * CHUNK_SIZE)

    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, config["NUM_ENVS"]))
    global_step = jnp.array(0, dtype=jnp.int32)
    carry = (agent_state, buffer_state, env_state, obs, key, global_step, episode_stats)

    print(f"[{progress_mode}] {total_iterations} chunks of {CHUNK_SIZE * config['NUM_ENVS']} steps")

    if progress_mode == "scan_outer":
        def log_cb(step, ret, length, loss):
            wandb.log({
                "charts/global_step": int(step),
                "charts/avg_episodic_return": float(ret),
                "charts/avg_episodic_length": float(length),
                "losses/td_loss": float(loss),
            }, step=int(step))

        def outer_step(carry, _):
            carry, (losses, epsilons) = jax.lax.scan(step_once, carry, None, length=CHUNK_SIZE)
            _, _, _, _, _, gstep, ep = carry
            avg_loss = jnp.sum(losses) / jnp.maximum(jnp.sum(losses != 0), 1)
            jax.debug.callback(
                log_cb, gstep, ep.returned_episode_returns.mean(),
                ep.returned_episode_lengths.mean(), avg_loss,
            )
            return carry, None

        @jax.jit
        def train(c):
            c, _ = _tqdx.scan(outer_step, c, None, length=total_iterations)
            return c

        start_time = time.time()
        carry = jax.block_until_ready(train(carry))
        wall = time.time() - start_time
        agent_state = carry[0]
        total_steps = int(carry[5])
        print(f"[scan_outer] {total_steps} steps in {wall:.1f}s incl. compile -> {int(total_steps/wall)} SPS (compile-inclusive)")
        eval_metrics = save_and_eval(total_steps, agent_state)
        wandb.finish()
        return eval_metrics

    scan_impl = _tqdx.scan if progress_mode == "tqdx_inner" else jax.lax.scan

    @partial(jax.jit, donate_argnums=(0,))
    def rollout_chunk(carry):
        return scan_impl(step_once, carry, None, length=CHUNK_SIZE)

    start_time = time.time()
    total_eval_time = 0.0
    iterator = range(1, total_iterations + 1)
    if progress_mode == "tqdm_outer":
        iterator = _tqdm(iterator, total=total_iterations, desc=run_name)

    for i in iterator:
        iteration_time_start = time.time()
        carry, (losses, epsilons) = rollout_chunk(carry)
        agent_state, buffer_state, env_state, obs, key, global_step, episode_stats = carry

        current_step = global_step.item()
        iteration_time = time.time() - iteration_time_start

        if config.get("EVAL_DURING_TRAIN", True) and (i % config.get("EVAL_EVERY", 10) == 0):
            eval_t0 = time.time()
            save_and_eval(current_step, agent_state)
            total_eval_time += time.time() - eval_t0

        wandb.log({
            "charts/avg_episodic_return": episode_stats.returned_episode_returns.mean().item(),
            "charts/avg_episodic_length": episode_stats.returned_episode_lengths.mean().item(),
            "charts/epsilon": epsilons[-1].item(),
            "charts/SPS": int(current_step / (time.time() - start_time - total_eval_time)),
            "charts/SPS_update": int(CHUNK_SIZE * config["NUM_ENVS"] / iteration_time),
            "losses/td_loss": float(jnp.sum(losses) / jnp.maximum(jnp.sum(losses != 0), 1)),
            "charts/global_step": current_step,
        }, step=current_step)

    final_sps = int(global_step.item() / (time.time() - start_time - total_eval_time))
    print(f"[{progress_mode}] final SPS (eval excluded): {final_sps}")
    eval_metrics = save_and_eval(config.get("TOTAL_TIMESTEPS", 10000000), agent_state)
    wandb.finish()
    return eval_metrics
