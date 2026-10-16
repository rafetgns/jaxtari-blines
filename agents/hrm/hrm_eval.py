from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxtari.environment import JaxEnvironment
from jaxtari.wrappers import JaxtariWrapper

from agents.run_utils import game_states

_NEG_INF = -1e9


def evaluate(
    option_params,
    meta_params,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    option_net: nn.Module,
    meta_net: nn.Module,
    options,
    num_rm_states: int,
    seed: int = 1,
    option_max_steps: int = 200,
):
    """Run `eval_episodes` episodes in parallel with the greedy meta-controller and option policies.

    The eval env's reward is the game's own (the RM wrapper passes it through in
    eval mode). Returns the per-episode returns and the env states of the first
    episode up to its end (used for video logging).
    """
    env: JaxEnvironment | JaxtariWrapper = make_env(env_id)()
    raw_dim = env.observation_space().shape[0] - num_rm_states
    availability = options.availability

    @jax.jit
    def wrapped_reset(key):
        """Reset the env and fix the observation shape to (1, obs_dim)."""
        obs, state = env.reset(key)
        return obs[None, ...], state

    @jax.jit
    def wrapped_step(state, action):
        """Step the env and fix the observation shape to (1, obs_dim)."""
        obs, state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return obs[None, ...], state, reward, done, info

    def u_of(obs):
        return jnp.argmax(obs[..., -num_rm_states:], axis=-1)

    @jax.jit
    def greedy_option(obs):
        q_vals = meta_net.apply(meta_params, obs)[0]  # (num_options,)
        return jnp.argmax(jnp.where(availability[u_of(obs[0])], q_vals, _NEG_INF))

    @jax.jit
    def greedy_action(obs, option):
        q_vals = option_net.apply(option_params, obs[..., :raw_dim])[0]  # (num_options, A)
        return jnp.argmax(q_vals[option], axis=-1)

    def step_fn(carry, _):
        obs, env_state, option, active, length = carry
        # commit a new option only where none is running
        option = jnp.where(active, option, jax.vmap(greedy_option)(obs))
        length = jnp.where(active, length, 0)

        actions = jax.vmap(greedy_action)(obs, option)
        obs, env_state, reward, done, info = jax.vmap(wrapped_step)(env_state, actions)
        length = length + 1
        terminate = info["option_terminate"] | done | (length >= option_max_steps)
        first_state = jax.tree.map(lambda x: x[0], env_state)
        return (obs, env_state, option, ~terminate, length), (first_state, done, reward)

    @jax.jit
    def scanned_step(carry):
        return jax.lax.scan(step_fn, carry, None, length=1000)

    obs, env_states = jax.vmap(wrapped_reset)(jax.random.split(jax.random.PRNGKey(seed), eval_episodes))
    carry = (
        obs,
        env_states,
        jnp.zeros(eval_episodes, jnp.int32),
        jnp.zeros(eval_episodes, bool),  # forces a meta choice on step 0
        jnp.zeros(eval_episodes, jnp.int32),
    )

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
