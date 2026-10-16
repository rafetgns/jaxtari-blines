"""Evaluation for the recurrent agent.

Same structure as agents/dqn/dqn_eval.py (greedy play, parameters in memory),
with what the LSTM forces: the policy carries a state from step to step (the
recurrent state, the previous action and reward, and for NGU and later stages
the previous curiosity reward and the episodic memory). That state and the
network live in agent57.single_run, which passes the policy as three functions,
the same ones its collection uses:

    init(n)                                      -> state for n episodes
    act(state, obs)                              -> (greedy actions, state)
    observe(state, actions, reward, obs, done)   -> state for the next step

prev_action = -1 marks the first step of an episode, which makes the network
start from a zero state (same convention as training).

The env is built with eval=True, so rewards are NOT clipped and a lost life is
not an episode end; this is the number comparable to the DQN and Rainbow evals.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from jaxtari.environment import JaxEnvironment
from jaxtari.wrappers import JaxtariWrapper

from agents.run_utils import game_states


def evaluate(
    policy: tuple,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    obs_shape: tuple,
    seed: int = 1,
):
    """Run `eval_episodes` episodes in parallel with the greedy policy.

    Returns the per-episode returns and the env states of the first episode
    up to its end (used for video logging).
    """
    init, act, observe = policy
    env: JaxEnvironment | JaxtariWrapper = make_env(env_id)()

    # reshape to the training obs_shape (drops the trailing channel of pixel observations)
    def reset_one(key):
        obs, state = env.reset(key)
        return obs.reshape(obs_shape), state

    def step_one(state, action):
        obs, state, reward, terminated, truncated, info = env.step(state, action)
        return obs.reshape(obs_shape), state, reward.astype(jnp.float32), jnp.logical_or(terminated, truncated)

    def step_fn(carry, _):
        obs, env_state, policy_state = carry
        actions, policy_state = act(policy_state, obs)
        obs, env_state, reward, done = jax.vmap(step_one)(env_state, actions)
        policy_state = observe(policy_state, actions, reward, obs, done)
        first_state = jax.tree.map(lambda x: x[0], env_state)
        return (obs, env_state, policy_state), (first_state, done, reward)

    @jax.jit
    def scanned_step(carry):
        return jax.lax.scan(step_fn, carry, None, length=1000)

    obs, env_states = jax.vmap(reset_one)(jax.random.split(jax.random.PRNGKey(seed), eval_episodes))
    carry = (obs, env_states, init(eval_episodes))

    first_states_chunks, done_chunks, reward_chunks = [], [], []
    done_ever = jnp.zeros(eval_episodes, dtype=jnp.bool_)
    while not jnp.all(done_ever):
        carry, (first_states, dones, rewards) = scanned_step(carry)
        first_states_chunks.append(first_states)
        done_chunks.append(dones)
        reward_chunks.append(rewards)
        done_ever = done_ever | jnp.any(dones, axis=0)

    first_states_history = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *first_states_chunks)
    dones = jnp.concatenate(done_chunks, axis=0)
    rewards = jnp.concatenate(reward_chunks, axis=0)

    # only count rewards up to and including each episode's first done
    first_done = jnp.argmax(dones, axis=0)
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
    episodic_returns = jnp.sum(rewards * (1 - mask_after_first_done), axis=0)
    print(
        f"Evaluated {eval_episodes} episodes, mean return: {episodic_returns.mean():.2f}, "
        f"std return: {episodic_returns.std():.2f}"
    )

    # the first episode's game states, for the eval video
    env_states_until_done = jax.tree.map(
        lambda x: x[: int(first_done[0]) + 1],
        game_states(first_states_history),
    )
    return episodic_returns, env_states_until_done
