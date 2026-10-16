"""Hierarchy of Reward Machines (HRM), Toro Icarte et al., ICML 2018.

A meta-controller picks an option at the current RM state; options are the RM
edges marked as subgoals ("option": True in a game's TRANSITIONS, see
reward_machines/options.py), and edges with the same guard share one option
policy. All option policies are heads of one Double DQN trained off-policy on
every step with binary pseudo-rewards (1 when the option's formula fires); the
meta-controller is SMDP Q-learning on the RM reward collected while an option
runs. Object-centric observations only. Evaluation scores the game's reward.

From PR #31 (agents/double_dqn_hrm), in the structure of agents/dqn/dqn.py.
"""

import os
import random
import time
from functools import partial
from typing import Any

import flashbax as fbx
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import struct
from flax.training.train_state import TrainState
import jaxtari
from jaxtari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
)
from rtpt import RTPT

from agents.hrm.hrm_eval import evaluate
from agents.run_utils import eval_mod_configs, plan_chunks, rm_metrics
from reward_machines.options import OptionSpec, build_options
from reward_machines.rm_registry import GAME_RM_REGISTRY, wrap_reward_machine

_NEG_INF = -1e9


def make_env(env_id, mods=[], options: OptionSpec = None, eval=False, use_shaping=False, gamma=0.99):
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
        # no CRM: HRM builds its own transitions from the option signals
        env = wrap_reward_machine(env, env_id, eval=eval, options=options, use_shaping=use_shaping, gamma=gamma)
        return LogWrapper(env)

    return thunk


@struct.dataclass
class OptionTimeStep:
    """One primitive step, usable to update every option head."""

    obs: jnp.ndarray
    action: jnp.ndarray
    option_rewards: jnp.ndarray
    next_obs: jnp.ndarray
    env_done: jnp.ndarray


@struct.dataclass
class MetaTimeStep:
    """One SMDP step: the whole execution of a single option."""

    obs: jnp.ndarray
    option: jnp.ndarray
    ret: jnp.ndarray
    next_obs: jnp.ndarray
    discount: jnp.ndarray
    done: jnp.ndarray


@struct.dataclass
class OptionExecState:
    """Per-env bookkeeping for the option currently in flight."""

    option: jnp.ndarray
    active: jnp.ndarray
    start_obs: jnp.ndarray
    ret: jnp.ndarray
    discount: jnp.ndarray
    length: jnp.ndarray


class OptionQNetwork(nn.Module):
    """Q-values of every option policy: (..., obs) -> (..., num_options, actions)."""

    num_options: int
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        leading = x.shape[:-1]
        x = nn.Dense(512, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        q = nn.Dense(self.num_options * self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
        return q.reshape(*leading, self.num_options, self.action_dim)


class MetaQNetwork(nn.Module):
    """Q-values of the options at the (observation, RM state): (..., obs + one-hot) -> (..., num_options)."""

    num_options: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(512, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        return nn.Dense(self.num_options, kernel_init=nn.initializers.orthogonal(0.01))(x)


@struct.dataclass
class RingBufferState:
    data: Any
    ptr: jnp.ndarray


class MaskedRingBuffer:
    """Uniform replay that only appends the rows marked valid (meta steps end at different times)."""

    def __init__(self, capacity: int, sample_batch_size: int, min_length: int):
        self.capacity = capacity
        self.sample_batch_size = sample_batch_size
        self.min_length = min_length

    def init(self, item: Any) -> RingBufferState:
        """`item` is a pytree of per-row shapes (no leading batch dim)."""
        data = jax.tree.map(lambda x: jnp.zeros((self.capacity + 1,) + x.shape, x.dtype), item)
        return RingBufferState(data=data, ptr=jnp.int32(0))

    def add(self, state: RingBufferState, batch: Any, valid: jnp.ndarray) -> RingBufferState:
        """Append the rows of `batch` where `valid` is True (others go to a scratch row)."""
        offsets = jnp.cumsum(valid) - 1
        dest = jnp.where(valid, (state.ptr + offsets) % self.capacity, self.capacity)
        data = jax.tree.map(lambda buf, x: buf.at[dest].set(x), state.data, batch)
        return state.replace(data=data, ptr=state.ptr + jnp.sum(valid).astype(jnp.int32))

    def sample(self, state: RingBufferState, key) -> Any:
        size = jnp.minimum(state.ptr, self.capacity)
        idx = jax.random.randint(key, (self.sample_batch_size,), 0, jnp.maximum(size, 1))
        return jax.tree.map(lambda buf: buf[idx], state.data)

    def can_sample(self, state: RingBufferState) -> jnp.ndarray:
        return state.ptr >= self.min_length


class HRMTrainState(TrainState):
    target_params: flax.core.FrozenDict


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    assert not config.get("PIXEL_BASED", False), "HRM needs object-centric observations (PIXEL_BASED: False)"
    env_id = config["ENV_ID"]
    if env_id not in GAME_RM_REGISTRY:
        raise ValueError(f"no reward machine for '{env_id}'; available: {sorted(GAME_RM_REGISTRY)}")
    use_shaping = bool(config.get("RM_SHAPING", False))
    gamma = config["GAMMA"]
    n_envs = config["NUM_ENVS"]
    k_max = config["OPTION_MAX_STEPS"]

    run_name = f"{env_id}_{config['EXP_NAME']}_oc{'_shaped' if use_shaping else ''}_{config['SEED']}"
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

    # --- reward machine + option structure (Python time, static) ---
    options = build_options(GAME_RM_REGISTRY[env_id]())
    print(options.describe())
    num_options = options.num_options
    availability = options.availability  # (num_states, num_options)

    env_kwargs = dict(options=options, use_shaping=use_shaping, gamma=gamma)
    env = make_env(env_id, list(config.get("TRAIN_MODS") or []), **env_kwargs)()
    num_rm_states = env._env.rm.num_states  # LogWrapper -> RewardMachineWrapper
    aug_dim = env.observation_space().shape[0]
    raw_dim = aug_dim - num_rm_states
    n_actions = env.action_space().n

    @jax.jit
    def vmap_reset(k):
        return jax.vmap(env.reset)(k)

    @jax.jit
    def vmap_step(state, action):
        obs, state, reward, term, trunc, info = jax.vmap(env.step)(state, action)
        return obs, state, reward, term | trunc, info

    # same budget rule for every agent: whole jitted chunks within TOTAL_TIMESTEPS
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(config["TOTAL_TIMESTEPS"], n_envs, config.get("SCAN_STEPS", 50))
    num_updates = scan_steps * num_chunks

    # --- learning rates ---
    warmup_iters = config["LEARNING_STARTS"] // n_envs
    num_grad_updates = max(1, (num_updates - warmup_iters) // config["TRAIN_FREQUENCY"])

    def linear_schedule(base, steps_per_update):
        return lambda count: base * (1.0 - count / (num_grad_updates * steps_per_update))

    anneal = config.get("ANNEAL_LR", False)
    option_lr = linear_schedule(config["LEARNING_RATE"], config["GRADIENT_STEPS"]) if anneal else config["LEARNING_RATE"]
    meta_lr = linear_schedule(config["META_LEARNING_RATE"], 1) if anneal else config["META_LEARNING_RATE"]

    # --- networks ---
    option_net = OptionQNetwork(num_options=num_options, action_dim=n_actions)
    meta_net = MetaQNetwork(num_options=num_options)

    # same root split as dqn/c51/pqn, so env resets for a given seed match across agents
    key, init_key = jax.random.split(key, 2)
    k_opt, k_meta = jax.random.split(init_key)

    def make_ts(net, init_key, dim, lr):
        params = net.init(init_key, jnp.zeros(dim))
        return HRMTrainState.create(
            apply_fn=net.apply,
            params=params,
            target_params=jax.tree.map(jnp.copy, params),
            tx=optax.chain(optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 10.0)), optax.adam(lr)),
        )

    option_ts = make_ts(option_net, k_opt, raw_dim, option_lr)  # options see the observation without the RM state
    meta_ts = make_ts(meta_net, k_meta, aug_dim, meta_lr)

    # --- buffers ---
    # option transitions: exactly one per env per step
    option_buffer = fbx.make_item_buffer(
        max_length=config["BUFFER_SIZE"],
        min_length=config["BATCH_SIZE"],
        sample_batch_size=config["BATCH_SIZE"],
        add_sequences=False,
        add_batches=True,
    )
    option_buffer = option_buffer.replace(
        init=jax.jit(option_buffer.init),
        add=jax.jit(option_buffer.add, donate_argnums=0),
        sample=jax.jit(option_buffer.sample),
        can_sample=jax.jit(option_buffer.can_sample),
    )
    option_bs = option_buffer.init(
        OptionTimeStep(
            obs=jnp.zeros(raw_dim, jnp.float32), action=jnp.int32(0),
            option_rewards=jnp.zeros(num_options, jnp.float32),
            next_obs=jnp.zeros(raw_dim, jnp.float32), env_done=jnp.bool_(False),
        )
    )
    # meta transitions: only when an option ends
    meta_buffer = MaskedRingBuffer(
        capacity=config["META_BUFFER_SIZE"],
        sample_batch_size=config["META_BATCH_SIZE"],
        min_length=config["META_BATCH_SIZE"],
    )
    meta_bs = meta_buffer.init(
        MetaTimeStep(
            obs=jnp.zeros(aug_dim, jnp.float32), option=jnp.int32(0),
            ret=jnp.float32(0), next_obs=jnp.zeros(aug_dim, jnp.float32),
            discount=jnp.float32(0), done=jnp.bool_(False),
        )
    )
    exec_state = OptionExecState(
        option=jnp.zeros(n_envs, jnp.int32),
        active=jnp.zeros(n_envs, bool),  # forces a meta choice on step 0
        start_obs=jnp.zeros((n_envs, aug_dim), jnp.float32),
        ret=jnp.zeros(n_envs, jnp.float32),
        discount=jnp.ones(n_envs, jnp.float32),
        length=jnp.zeros(n_envs, jnp.int32),
    )
    obs, env_state = vmap_reset(jax.random.split(key, n_envs))

    def u_of(aug_obs):
        """The RM state, from the observation's one-hot tail."""
        return jnp.argmax(aug_obs[..., -num_rm_states:], axis=-1)

    def option_epsilon(global_step):
        return jnp.interp(
            global_step,
            jnp.array([0, config.get("EXPLORATION_FRACTION", 0.3) * config["TOTAL_TIMESTEPS"]]),
            jnp.array([config.get("START_E", 1.0), config.get("END_E", 0.02)]),
        )

    def meta_epsilon(meta_steps):
        # annealed over meta steps (finished options), not env steps
        return jnp.interp(
            meta_steps,
            jnp.array([0, config["META_EXPLORATION_STEPS"]]),
            jnp.array([config.get("META_START_E", 1.0), config.get("META_END_E", 0.1)]),
        )

    def eps_greedy(rng, q, eps):
        rng_a, rng_e = jax.random.split(rng)
        greedy = jnp.argmax(q, axis=-1)
        rand = jax.random.randint(rng_a, greedy.shape, 0, q.shape[-1])
        return jnp.where(jax.random.uniform(rng_e, greedy.shape) < eps, rand, greedy)

    def eps_greedy_masked(rng, q, mask, eps):
        rng_g, rng_e = jax.random.split(rng)
        greedy = jnp.argmax(jnp.where(mask, q, _NEG_INF), axis=-1)
        rand = jnp.argmax(jnp.where(mask, jax.random.gumbel(rng_g, q.shape), _NEG_INF), axis=-1)
        return jnp.where(jax.random.uniform(rng_e, greedy.shape) < eps, rand, greedy)

    def _option_learn(ts, buf_state, rng):
        batch = option_buffer.sample(buf_state, rng).experience
        # Double DQN per option head
        a_star = jnp.argmax(option_net.apply(ts.params, batch.next_obs), axis=-1)  # (B, O)
        q_next = jnp.take_along_axis(option_net.apply(ts.target_params, batch.next_obs), a_star[..., None], -1).squeeze(-1)
        # an option's own task terminates when its formula becomes true
        done_o = (batch.option_rewards > 0) | batch.env_done[:, None]
        target = batch.option_rewards + (1.0 - done_o) * gamma * q_next  # (B, O)

        def _loss(params):
            q = option_net.apply(params, batch.obs)  # (B, O, A)
            act = jnp.broadcast_to(batch.action[:, None, None], q.shape[:-1] + (1,))
            chosen = jnp.take_along_axis(q, act, -1).squeeze(-1)  # (B, O)
            return jnp.mean(optax.huber_loss(chosen, target, delta=1.0))

        loss, grads = jax.value_and_grad(_loss)(ts.params)
        return ts.apply_gradients(grads=grads), loss

    def _option_learn_k(ts, buf_state, rng):
        def _grad_step(carry, _):
            ts, rng = carry
            rng, step_rng = jax.random.split(rng)
            ts, loss = _option_learn(ts, buf_state, step_rng)
            return (ts, rng), loss

        (ts, _), losses = jax.lax.scan(_grad_step, (ts, rng), None, config["GRADIENT_STEPS"])
        return ts, losses.mean()

    def _meta_learn(ts, buf_state, rng):
        """SMDP Q-learning: y = sum_k gamma^k r_{t+k} + gamma^K max_{o' in avail(u')} Q^-(s', u', o')."""
        batch = meta_buffer.sample(buf_state, rng)
        mask_next = availability[u_of(batch.next_obs)]  # (B, O)
        o_star = jnp.argmax(jnp.where(mask_next, meta_net.apply(ts.params, batch.next_obs), _NEG_INF), axis=-1)
        q_next = jnp.take_along_axis(meta_net.apply(ts.target_params, batch.next_obs), o_star[:, None], -1).squeeze(-1)
        target = batch.ret + (1.0 - batch.done) * batch.discount * q_next

        def _loss(params):
            q = meta_net.apply(params, batch.obs)
            chosen = jnp.take_along_axis(q, batch.option[:, None], -1).squeeze(-1)
            return jnp.mean(optax.huber_loss(chosen, target, delta=1.0))

        loss, grads = jax.value_and_grad(_loss)(ts.params)
        return ts.apply_gradients(grads=grads), loss

    def hrm_step(carry):
        option_ts, meta_ts, option_bs, meta_bs, env_state, last_obs, exec_state, rng, meta_steps, global_step = carry
        raw_obs = last_obs[..., :raw_dim]
        rng, rng_m, rng_a, rng_lo, rng_lm = jax.random.split(rng, 5)

        # 1. the meta-controller commits a new option wherever none is in flight
        mask = availability[u_of(last_obs)]
        proposed = eps_greedy_masked(rng_m, meta_net.apply(meta_ts.params, last_obs), mask, meta_epsilon(meta_steps))
        started = ~exec_state.active
        option = jnp.where(started, proposed, exec_state.option)
        start_obs = jnp.where(started[:, None], last_obs, exec_state.start_obs)
        ret = jnp.where(started, 0.0, exec_state.ret)
        discount = jnp.where(started, 1.0, exec_state.discount)
        length = jnp.where(started, 0, exec_state.length)

        # 2. the active option's policy picks a primitive action
        q_opt = jnp.take_along_axis(option_net.apply(option_ts.params, raw_obs), option[:, None, None], axis=1).squeeze(1)
        action = eps_greedy(rng_a, q_opt, option_epsilon(global_step))
        obs, env_state, reward, done, info = vmap_step(env_state, action)
        global_step = global_step + n_envs

        # 3. option replay: one row per env, every head gets its pseudo-reward
        option_bs = option_buffer.add(option_bs, OptionTimeStep(
            obs=raw_obs, action=action, option_rewards=info["option_rewards"],
            next_obs=obs[..., :raw_dim], env_done=done,
        ))

        # 4. the discounted SMDP return of the running option
        ret = ret + discount * reward
        discount = discount * gamma
        length = length + 1

        # 5. terminate on subgoal, RM state change, episode end or timeout (the timeout is not in the
        #    paper, but without it a stuck option never releases control and yields no meta sample)
        terminate = info["option_terminate"] | done | (length >= k_max) | (u_of(obs) != u_of(last_obs))
        meta_bs = meta_buffer.add(
            meta_bs,
            MetaTimeStep(obs=start_obs, option=option, ret=ret, next_obs=obs, discount=discount, done=done),
            valid=terminate,
        )
        meta_steps = meta_steps + jnp.sum(terminate).astype(jnp.int32)
        exec_state = OptionExecState(option=option, active=~terminate, start_obs=start_obs,
                                     ret=ret, discount=discount, length=length)

        # 6. gradient steps
        env_iters = global_step // n_envs
        can_learn = (
            option_buffer.can_sample(option_bs)
            & (global_step > config["LEARNING_STARTS"])
            & (env_iters % config["TRAIN_FREQUENCY"] == 0)
        )
        option_ts, option_loss = jax.lax.cond(
            can_learn, lambda ts, r: _option_learn_k(ts, option_bs, r),
            lambda ts, r: (ts, jnp.float32(0)), option_ts, rng_lo,
        )
        meta_ts, meta_loss = jax.lax.cond(
            can_learn & meta_buffer.can_sample(meta_bs),
            lambda ts, r: _meta_learn(ts, meta_bs, r),
            lambda ts, r: (ts, jnp.float32(0)), meta_ts, rng_lm,
        )

        def _sync(ts, frequency):
            return jax.lax.cond(
                (global_step % frequency) < n_envs,
                lambda s: s.replace(target_params=optax.incremental_update(s.params, s.target_params, config["TAU"])),
                lambda s: s, ts,
            )

        option_ts = _sync(option_ts, config["TARGET_NETWORK_FREQUENCY"])
        meta_ts = _sync(meta_ts, config["META_TARGET_NETWORK_FREQUENCY"])

        metrics = {
            "option_loss": option_loss,
            "meta_loss": meta_loss,
            "rm_reward": info["rm_reward"].mean(),
            "option_len": jnp.where(terminate, length, 0).sum() / jnp.maximum(jnp.sum(terminate), 1),
            "option_hist": jnp.sum(jax.nn.one_hot(option, num_options), axis=0),
            "subgoal_hits": info["option_rewards"].sum(axis=0),
        }
        carry = (option_ts, meta_ts, option_bs, meta_bs, env_state, obs, exec_state, rng, meta_steps, global_step)
        return carry, (info, metrics)

    def save_and_eval(params, step_count):
        option_params, meta_params = params
        if config.get("SAVE_PATH") is not None:
            save_dir = os.path.join(config["SAVE_PATH"], run_name)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model')
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, option_params, meta_params]))
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")
        metrics = {}
        for mods_cfg, mod_label in eval_mod_configs(config):
            # the eval env scores the game reward; the RM provides the state and the option signals
            episodic_returns, env_states = evaluate(
                option_params,
                meta_params,
                partial(make_env, mods=mods_cfg, eval=True, **env_kwargs),
                env_id,
                eval_episodes=10,
                option_net=option_net,
                meta_net=meta_net,
                options=options,
                num_rm_states=num_rm_states,
                seed=config["SEED"] + 42,  # use a different seed for evaluation
                option_max_steps=k_max,
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

    print("[hrm] start compile...")
    start_compile = time.perf_counter()
    carry = (option_ts, meta_ts, option_bs, meta_bs, env_state, obs, exec_state, key,
             jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))

    def train_chunk(carry):
        return jax.lax.scan(lambda c, _: hrm_step(c), carry, None, length=scan_steps)

    # donate the carry so XLA reuses the replay buffers' memory; lower/compile AOT so nothing is donated before the loop
    compiled = jax.jit(train_chunk, donate_argnums=(0,)).lower(carry).compile()
    print(f"[hrm] compilation time: {time.perf_counter() - start_compile:.2f}s")

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[hrm] starting training: {num_chunks} chunks x {steps_per_chunk} env steps")
    for chunk in range(num_chunks):
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
            save_and_eval((carry[0].params, carry[1].params), int(carry[-1]))

        chunk_start = time.perf_counter()
        carry, (infos, m) = compiled(carry)
        global_step = int(carry[-1])
        metrics = {
            # the game's return (LogWrapper logs info["env_reward"]); the RM's side comes from rm_metrics
            "charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
            "charts/avg_episodic_length": float(infos["returned_episode_lengths"][-1].mean()),
            "charts/rm_reward_per_step": float(m["rm_reward"].mean()),
            "charts/mean_option_length": float(m["option_len"].mean()),
            "losses/option_loss": float(m["option_loss"][-1]),
            "losses/meta_loss": float(m["meta_loss"][-1]),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        hist = np.asarray(m["option_hist"].sum(axis=0))
        hits = np.asarray(m["subgoal_hits"].sum(axis=0))
        for i, name in enumerate(options.names):
            metrics[f"options/selected_{i}_{name}"] = float(hist[i])
            metrics[f"options/achieved_{i}_{name}"] = float(hits[i])
        metrics.update(rm_metrics(infos))
        wandb.log(metrics, step=global_step)
        print(
            f"[hrm] chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
            f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
            f"option_len {metrics['charts/mean_option_length']:.1f} | "
            f"option_loss {metrics['losses/option_loss']:.3g} | meta_loss {metrics['losses/meta_loss']:.3g} | "
            f"SPS {metrics['charts/SPS']}"
        )

    eval_metrics = save_and_eval((carry[0].params, carry[1].params), global_step + 1)
    wandb.finish()
    return eval_metrics
