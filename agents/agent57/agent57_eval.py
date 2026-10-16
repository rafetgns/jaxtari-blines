"""Evaluation for the recurrent agent.

Same structure as agents/dqn/dqn_eval.py, with the three changes the LSTM forces:
  - the recurrent state is threaded through the episode
  - prev_action / prev_reward are fed back into the network
  - prev_action = -1 marks the first step of an episode, which makes the
    network start from a zero state (same convention as training)

The env is built with eval=True, so rewards are NOT clipped and a lost life is
not an episode end. Training returns use clipped rewards, so for games whose
rewards are not just +-1 the two numbers differ; this is the one comparable to
the DQN and Rainbow eval numbers.

WHAT THE LSTM CHANGES, COMPARED TO dqn_eval.py
    1. The recurrent state (carry) is threaded from step to step, so the agent
       remembers what happened earlier in the episode.
    2. The previous action (one-hot) and previous reward are fed back into the
       network, as R2D2's architecture requires.
    3. prev_action = -1 marks the first step of an episode. The network reads
       that as "no previous action" and wipes its memory. This is the same
       convention the training collection loop uses, so an episode start looks
       identical in both places.
 
RETURNS
    episodic_returns        (eval_episodes,) unclipped return of each episode
    env_states_until_done   states of episode 0 up to its first done, for video
                            capture; None if the env does not expose them.
"""

from typing import Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxatari.environment import JaxEnvironment
from jaxatari.wrappers import JaxatariWrapper


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    Model: nn.Module,
    network_kwargs: dict,
    epsilon: float = 0.05,      # same as dqn_eval's default, so the Atari protocol matches
    seed: int = 1,
    max_steps: int = 200_000,
    intrinsic=None,
):
#Play `eval_episodes` full games with the trained weights and score them.
   # ---- 1. the evaluation environment ------------------------------------
   # eval=True was bound by the caller: rewards are real game points and only a genuine game over (or after 30 minutes) ends an episode.
    env: JaxEnvironment | JaxatariWrapper = make_env(env_id)()
    action_dim = env.action_space().n
    key = jax.random.PRNGKey(seed)
    # ---- 2. single-episode helpers, then vectorised over episodes ----------
    # squeeze() drops the trailing channel dimension of pixel observations
    # ((4,84,84,1) -> (4,84,84)), matching single_run's obs_shape[:-1].

    def reset_one(key):
        obs, state = env.reset(key)
        return obs.squeeze(), state

    def step_one(state, action):
        obs, state, reward, terminated, truncated, info = env.step(state, action)
        return obs.squeeze(), state, reward, jnp.logical_or(terminated, truncated), info

    # vmap turns one episode into `eval_episodes` episodes running side by side,

    vmap_reset = jax.vmap(reset_one)
    vmap_step = jax.vmap(step_one)
    # ---- 3. rebuild the network and load the trained weights ---------------
    # action_dim comes from the env

    network = Model(action_dim=action_dim, **network_kwargs)

    # [NGU] an NGU network also takes (arm, previous intrinsic reward). Evaluation
    # plays arm 0, the exploitative arm (beta = 0), with the previous intrinsic
    # reward fed as 0: no episodic memory or RND runs at eval time.
    # [NGU] an NGU network also takes (arm, previous intrinsic reward). Evaluation
    # plays arm 0, the exploitative arm. The previous intrinsic reward must be the
    # real one, computed as in training: fed as a constant 0 instead, arm 0 falls
    # from about -10 to -20 on Pong, because the network treats that input as part
    # of the situation. `intrinsic` = (fn, init_memory), provided by single_run:
    #   fn(memory, obs, done) -> (r_int, memory)     same pipeline as collection
    def arm_inputs(n, prev_int):
        if network.num_arms > 0:
            return (jnp.zeros((n,), jnp.int32), prev_int)
        return ()

    key, net_key, reset_key = jax.random.split(key, 3)
    dummy_obs, _ = reset_one(reset_key)

    # the network is initialised with random weights first and then overwritten from the file.
    params_template = network.init(
        net_key,
        Model.initial_carry(1, network_kwargs.get("hidden_size", 512)),
        dummy_obs[None],
        jnp.zeros((1,), jnp.int32),
        jnp.zeros((1,), jnp.float32),
        *arm_inputs(1, jnp.zeros((1,), jnp.float32)),
    )


    # [SPLIT_Q] Peek at the saved file to tell an r2d2/ngu checkpoint
    # ([config, params]) apart from a split_q one ({"config":..., "params_e":
    # ..., "params_i":...}), WITHOUT changing how r2d2/ngu checkpoints load
    # (that exact original line still runs, unchanged, in the else branch).
    with open(model_path, "rb") as f:
        _raw_bytes = f.read()
    _peek = flax.serialization.from_bytes(None, _raw_bytes)
    split_q = isinstance(_peek, dict) and _peek.get("config", {}).get("STAGE") in ("split_q", "agent57")

    if split_q:
        # Deferred import: agent57.py imports `evaluate` from this module at
        # LOAD time, so importing it at the top of this file would be
        # circular. By the time evaluate() is actually CALLED (end of
        # single_run), agent57.py has finished executing top to bottom.
        from agents.agent57.agent57 import arm_schedule

        saved_config = _peek["config"]
        params_e = flax.serialization.from_state_dict(params_template, _peek["params_e"])
        params_i = flax.serialization.from_state_dict(params_template, _peek["params_i"])
        arm_betas, _ = arm_schedule(
            saved_config["NUM_ARMS"], saved_config["BETA_MAX"],
            saved_config["GAMMA_MAX"], saved_config["GAMMA_MIN"])
        beta_0 = arm_betas[0]   # arm 0 is the evaluator's existing convention
    else:
        params = params_template
        with open(model_path, "rb") as f:
            (_, params) = flax.serialization.from_bytes((None, params), f.read())
    
    
     # ---- 4. one step of play, for all episodes at once ---------------------
    def step_fn(carry, _):
        if split_q:
            obs, env_state, lstm_e, lstm_i, prev_action, prev_reward, prev_int, memory, rng = carry
        else:
            obs, env_state, lstm, prev_action, prev_reward, prev_int, memory, rng = carry
        rng, action_rng, explore_rng = jax.random.split(rng, 3)
        
        # the LSTM state is carried forward; the network wipes it itself wherever
        # prev_action is -1

        if split_q:
            extra = arm_inputs(obs.shape[0], prev_int)   # identical inputs to both networks
            lstm_e, q_e = network.apply(params_e, lstm_e, obs, prev_action, prev_reward, *extra)
            lstm_i, q_i = network.apply(params_i, lstm_i, obs, prev_action, prev_reward, *extra)
            q_values = q_e + beta_0 * q_i   # raw-output combination, no h_inv/h
        else:
            lstm, q_values = network.apply(params, lstm, obs, prev_action, prev_reward,
                                           *arm_inputs(obs.shape[0], prev_int))
        

        greedy = q_values.argmax(axis=-1)
        random_actions = jax.random.randint(action_rng, greedy.shape, 0, action_dim)
        explore = jax.random.uniform(explore_rng, greedy.shape) < epsilon
        actions = jnp.where(explore, random_actions, greedy)

        obs, env_state, reward, done, _ = vmap_step(env_state, actions)
        reward = reward.astype(jnp.float32)

        # a finished episode starts the next one with no memory
        prev_action = jnp.where(done, -1, actions)
        prev_reward = jnp.where(done, 0.0, reward)

        # [NGU] intrinsic reward of the state reached, exactly as in training
        if intrinsic is not None:
            r_int, memory = intrinsic[0](memory, obs, done)
            prev_int = jnp.where(done, 0.0, r_int)

        # env_state holds all episodes; x[0] takes episode 0's slice. Collected
        # over the scan these frames become the video of one full episode,
        # rendered for the report (CAPTURE_VIDEO), same as dqn_eval.py does.
        first_states = jax.tree.map(lambda x: x[0], env_state)
        
        if split_q:
            return (obs, env_state, lstm_e, lstm_i, prev_action, prev_reward, prev_int, memory, rng), \
                   (first_states, done, reward)
        else:
            return (obs, env_state, lstm, prev_action, prev_reward, prev_int, memory, rng), \
                   (first_states, done, reward)

    @jax.jit
    def scanned_steps(carry):
        return jax.lax.scan(step_fn, carry, None, length=1000)


    # ---- 5. run until every episode has finished at least once -------------
    key, rng = jax.random.split(key)
    reset_keys = jax.random.split(rng, eval_episodes)
    obs, env_states = vmap_reset(reset_keys)

    if split_q:
        carry = (
            obs, env_states,
            Model.initial_carry(eval_episodes, network_kwargs.get("hidden_size", 512)),   # lstm_e
            Model.initial_carry(eval_episodes, network_kwargs.get("hidden_size", 512)),   # lstm_i
            jnp.full((eval_episodes,), -1, jnp.int32), jnp.zeros((eval_episodes,), jnp.float32),
            jnp.zeros((eval_episodes,), jnp.float32),
            intrinsic[1](eval_episodes) if intrinsic is not None else (),
            key,
        )
    else:
        carry = (
            obs, env_states, Model.initial_carry(eval_episodes, network_kwargs.get("hidden_size", 512)),
            jnp.full((eval_episodes,), -1, jnp.int32), jnp.zeros((eval_episodes,), jnp.float32),
            jnp.zeros((eval_episodes,), jnp.float32),
            intrinsic[1](eval_episodes) if intrinsic is not None else (),
            key,
        )

    all_first_states, all_dones, all_rewards = [], [], []
    done_ever = jnp.zeros(eval_episodes, dtype=jnp.bool_)
    steps = 0

    while not jnp.all(done_ever) and steps < max_steps:
        carry, (first_states_chunk, dones_chunk, rewards_chunk) = scanned_steps(carry)
        all_first_states.append(first_states_chunk)
        all_dones.append(dones_chunk)
        all_rewards.append(rewards_chunk)
        done_ever = done_ever | jnp.any(dones_chunk, axis=0)
        steps += 1000

    first_states_history = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *all_first_states)
    dones = jnp.concatenate(all_dones, axis=0)
    rewards = jnp.concatenate(all_rewards, axis=0)

     # ---- 6. score the FIRST episode of each parallel run -------------------
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
    episodic_returns = jnp.sum(rewards * (1 - mask_after_first_done), axis=0)

    finished = int(jnp.sum(done_ever))
    print(
        f"[agent57_eval] {finished}/{eval_episodes} episodes finished | "
        f"mean return {episodic_returns.mean():.2f} | std {episodic_returns.std():.2f}"
    )

     # ---- 7. states of episode 0, for the report video ----------------------
    first_done = jnp.argmax(dones, axis=0)
    try:
        states = first_states_history.atari_state.atari_state.env_state
        env_states_until_done = jax.tree.map(lambda x: x[: first_done[0] + 1], states)
    except AttributeError:
        # wrapper chain does not expose that nesting; skip the video rather than
        # failing an evaluation that otherwise succeeded
        env_states_until_done = None

    return episodic_returns, env_states_until_done