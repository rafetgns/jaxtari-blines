"""Double DQN on reward machines, with counterfactual experiences (CRM).

Toro Icarte et al., "Using Reward Machines for High-Level Task Specification
and Decomposition in Reinforcement Learning", ICML 2018. The agent sees the
object-centric observation with the RM state appended and learns from the RM
reward (reward_machines/). With USE_CRM every env step is stored once per RM
state, relabelled with the reward and next state that RM state would have
produced, so the Q-function learns the whole machine from one interaction.
Evaluation scores the game's own reward.

Based on the Double DQN of PR #31 (after PureJaxRL's DQN,
https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/dqn.py), in the
structure of agents/dqn/dqn.py.
"""
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
import jaxtari
from jaxtari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
)
from agents.dqn.dqn_eval import evaluate
from agents.run_utils import eval_mod_configs, plan_chunks, rm_metrics
from reward_machines.reward_machine_wrapper import TimeStep
from reward_machines.rm_registry import GAME_RM_REGISTRY, wrap_reward_machine
from rtpt import RTPT


def make_env(env_id, mods=[], eval=False, use_crm=False, use_shaping=False, gamma=0.99):
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None
    if not eval and mods is not None and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxtari.make(env_id, mods=mods)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,
            first_fire=True,
            noop_max=30,
            full_action_space=False,
        )
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
        # CRM experiences are only needed while training
        env = wrap_reward_machine(env, env_id, eval=eval, use_crm=use_crm and not eval,
                                  use_shaping=use_shaping, gamma=gamma)
        return LogWrapper(env)
    return thunk


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)


class DQNTrainState(TrainState):
    target_params: flax.core.FrozenDict


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    assert not config.get("PIXEL_BASED", False), "RM_DQN needs object-centric observations (PIXEL_BASED: False)"
    env_id = config["ENV_ID"]
    if env_id not in GAME_RM_REGISTRY:
        raise ValueError(f"no reward machine for '{env_id}'; available: {sorted(GAME_RM_REGISTRY)}")
    use_crm = bool(config.get("USE_CRM", True))
    use_shaping = bool(config.get("RM_SHAPING", False))
    gamma = config.get("GAMMA", 0.99)

    run_name = f"{env_id}_{config['EXP_NAME']}_oc{'_crm' if use_crm else ''}{'_shaped' if use_shaping else ''}_{config['SEED']}"
    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
        mode=config.get("WANDB_MODE", "online"),
    )
    wandb.define_metric("*", step_metric="charts/global_step")

    # do not modify the seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    env_kwargs = dict(use_shaping=use_shaping, gamma=gamma)
    env = make_env(env_id, list(config.get("TRAIN_MODS") or []), False, use_crm=use_crm, **env_kwargs)()
    action_dim = env.action_space().n
    obs_dim = env.observation_space().shape[0]
    game_rm = env._env.rm  # LogWrapper -> RewardMachineWrapper
    num_transitions = len(game_rm.rewards)

    num_envs = config["NUM_ENVS"]
    gradient_steps = config.get("GRADIENT_STEPS", 32)
    batch_size = config.get("BATCH_SIZE", 512)
    total_timesteps = config["TOTAL_TIMESTEPS"]
    steps_per_update = config.get("TRAIN_FREQUENCY", 1) * num_envs

    @jax.jit
    def vmap_reset(rng):
        return jax.vmap(env.reset)(rng)

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        return next_obs, state, reward, jnp.logical_or(terminated, truncated), info

    # same budget rule for every agent: whole jitted chunks within TOTAL_TIMESTEPS
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(total_timesteps, steps_per_update, config.get("SCAN_STEPS", 50))
    num_updates = scan_steps * num_chunks

    # same root split as dqn/c51/pqn, so env resets for a given seed match across agents
    key, init_key = jax.random.split(key, 2)
    network = QNetwork(action_dim=action_dim)
    q_params = network.init(init_key, jnp.zeros((1, obs_dim)))

    def linear_schedule(count):
        # counts gradient steps; decays over the updates after LEARNING_STARTS
        learn_updates = max(1, num_updates - config["LEARNING_STARTS"] // steps_per_update) * gradient_steps
        return config["LEARNING_RATE"] * (1.0 - count / learn_updates)

    tx = optax.chain(
        optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 10.0)),
        optax.adam(learning_rate=linear_schedule if config.get("ANNEAL_LR", False) else config["LEARNING_RATE"]),
    )
    agent_state = DQNTrainState.create(
        apply_fn=network.apply,
        params=q_params,
        target_params=jax.tree.map(jnp.copy, q_params),
        tx=tx,
    )

    # explicit next_obs: CRM experiences are relabelled copies, not a trajectory
    replay_buffer = fbx.make_item_buffer(
        max_length=config.get("BUFFER_SIZE", 500_000),
        min_length=batch_size,
        sample_batch_size=batch_size,
        add_sequences=False,
        add_batches=True,
    )
    replay_buffer = replay_buffer.replace(
        init=jax.jit(replay_buffer.init),
        add=jax.jit(replay_buffer.add, donate_argnums=0),
        sample=jax.jit(replay_buffer.sample),
        can_sample=jax.jit(replay_buffer.can_sample),
    )
    obs, env_state = vmap_reset(jax.random.split(key, num_envs))
    buffer_state = replay_buffer.init(TimeStep(
        obs=jnp.zeros(obs_dim), action=jnp.zeros((), jnp.int32), reward=jnp.zeros(()),
        next_obs=jnp.zeros(obs_dim), done=jnp.zeros((), bool),
    ))

    def full_dqn_step(agent_state, buffer_state, env_state, obs, rng, global_step):
        def take_action(carry, _):
            agent_state, buffer_state, env_state, obs, global_step, rng = carry
            rng, action_rng, explore_rng = jax.random.split(rng, 3)
            epsilon = jnp.interp(
                global_step,
                jnp.array([0, config.get("EXPLORATION_FRACTION", 0.3) * total_timesteps]),
                jnp.array([config.get("START_E", 1.0), config.get("END_E", 0.05)]),
            )
            greedy_actions = agent_state.apply_fn(agent_state.params, obs).argmax(axis=-1)
            random_actions = jax.random.randint(action_rng, (num_envs,), 0, action_dim)
            explore_mask = jax.random.uniform(explore_rng, (num_envs,)) < epsilon
            actions = jnp.where(explore_mask, random_actions, greedy_actions)

            next_obs, next_env_state, rewards, next_done, info = vmap_step(env_state, actions)
            if use_crm:
                # (num_envs, num_rm_states, ...) -> one row per experience
                to_add = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), info.pop("experiences"))
            else:
                to_add = TimeStep(obs=obs, action=actions, reward=rewards, next_obs=next_obs, done=next_done)
            buffer_state = replay_buffer.add(buffer_state, to_add)
            info["rm_fired"] = jnp.sum(jax.nn.one_hot(info["rm_fired_idx"], num_transitions), axis=0)
            return (agent_state, buffer_state, next_env_state, next_obs, global_step + num_envs, rng), info

        (agent_state, buffer_state, next_env_state, next_obs, global_step, rng), infos = jax.lax.scan(
            take_action,
            (agent_state, buffer_state, env_state, obs, global_step, rng),
            None,
            length=config.get("TRAIN_FREQUENCY", 1),
        )

        def do_update(update_carry, _):
            u_state, u_key = update_carry
            u_key, sample_key = jax.random.split(u_key)
            batch = replay_buffer.sample(buffer_state, sample_key).experience

            # Double DQN: the online network picks the next action, the target network scores it
            next_actions = u_state.apply_fn(u_state.params, batch.next_obs).argmax(axis=-1)
            q_next = jnp.take_along_axis(
                u_state.apply_fn(u_state.target_params, batch.next_obs), next_actions[:, None], axis=-1
            ).squeeze(-1)
            target = jax.lax.stop_gradient(batch.reward + (1.0 - batch.done) * gamma * q_next)

            def q_loss_fn(params):
                q_pred = jnp.take_along_axis(u_state.apply_fn(params, batch.obs), batch.action[:, None], axis=-1).squeeze(-1)
                return jnp.mean(optax.huber_loss(q_pred, target, delta=1.0)), q_pred.mean()

            (loss, q_val), grads = jax.value_and_grad(q_loss_fn, has_aux=True)(u_state.params)
            return (u_state.apply_gradients(grads=grads), u_key), (loss, q_val)

        def scanned_update(carry):
            carry, (loss, qval) = jax.lax.scan(do_update, carry, None, length=gradient_steps)
            return carry, (loss.mean(), qval[-1])

        learning = replay_buffer.can_sample(buffer_state) & (global_step > config.get("LEARNING_STARTS", 20_000))
        (agent_state, rng), (loss, q_val) = jax.lax.cond(
            learning,
            scanned_update,
            lambda c: (c, (jnp.array(0.0), jnp.array(0.0))),
            (agent_state, rng),
        )
        update_target_flag = learning & ((global_step % config.get("TARGET_NETWORK_FREQUENCY", 24_576)) < steps_per_update)
        agent_state = agent_state.replace(target_params=jax.lax.cond(
            update_target_flag,
            lambda _: optax.incremental_update(agent_state.params, agent_state.target_params, config.get("TAU", 1.0)),
            lambda _: agent_state.target_params,
            None,
        ))
        return (agent_state, buffer_state, next_env_state, next_obs, rng, global_step), (infos, loss, q_val)

    def save_and_eval(q_params, step_count):
        if config.get("SAVE_PATH") is not None:
            save_dir = os.path.join(config["SAVE_PATH"], run_name)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model')
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, q_params]))
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")
        metrics = {}
        for mods_cfg, mod_label in eval_mod_configs(config):
            # the eval env scores the game reward; the RM only provides the state in the observation
            episodic_returns, env_states = evaluate(
                q_params,
                partial(make_env, mods=mods_cfg, eval=True, **env_kwargs),
                env_id,
                eval_episodes=10,
                Model=QNetwork,
                seed=config["SEED"] + 42,  # use a different seed for evaluation
            )
            metrics[mod_label] = float(np.mean(jax.device_get(episodic_returns)))
            wandb.log({f"eval/episodic_return_{mod_label}": metrics[mod_label]}, step=step_count)

            if config.get("CAPTURE_VIDEO", False):
                clean_renderer = jaxtari.make(env_id, mods=mods_cfg or None).renderer
                frames = jnp.transpose(jax.vmap(clean_renderer.render)(env_states), (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                wandb.log({f"eval/video_{mod_label}": video}, step=step_count)
                print(f"Video (eval) logged to wandb with {frames.shape[0]} frames ({mod_label}).")
        return metrics

    print("[rm_dqn] start compile...")
    start_compile = time.perf_counter()
    global_step = jnp.array(0, dtype=jnp.int32)
    carry = (agent_state, buffer_state, env_state, obs, key, global_step)

    def scanned_steps(carry):
        return jax.lax.scan(lambda c, _: full_dqn_step(*c), carry, None, length=scan_steps)

    # donate the carry so XLA reuses the replay buffer's memory; lower/compile AOT so nothing is donated before the loop
    compiled = jax.jit(scanned_steps, donate_argnums=(0,)).lower(carry).compile()
    print(f"[rm_dqn] compilation time: {time.perf_counter() - start_compile:.2f}s")

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[rm_dqn] starting training: {num_chunks} chunks x {steps_per_chunk} env steps")
    for chunk in range(num_chunks):
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
            save_and_eval(carry[0].params, int(carry[-1]))

        chunk_start = time.perf_counter()
        carry, (infos, loss, q_val) = compiled(carry)
        global_step = int(carry[-1])
        metrics = {
            # the game's return (LogWrapper logs info["env_reward"]); the RM's side comes from rm_metrics
            "charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
            "charts/avg_episodic_length": float(infos["returned_episode_lengths"][-1].mean()),
            "charts/rm_reward_per_step": float(infos["rm_reward"].mean()),
            "losses/td_loss": loss[-1].item(),
            "losses/q_values": q_val[-1].item(),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        fired = np.asarray(infos["rm_fired"].sum(axis=(0, 1)))  # (scan, train_freq, T) -> (T,)
        metrics.update({f"rm/transition_{i}": float(n) for i, n in enumerate(fired)})
        metrics.update(rm_metrics(infos))
        wandb.log(metrics, step=global_step)
        print(
            f"[rm_dqn] chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
            f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
            f"rm_return {metrics.get('charts/avg_episodic_rm_return', 0.0):.2f} | "
            f"td_loss {metrics['losses/td_loss']:.4f} | SPS {metrics['charts/SPS']}"
        )

    eval_metrics = save_and_eval(carry[0].params, global_step + 1)
    wandb.finish()
    return eval_metrics
