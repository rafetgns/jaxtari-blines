"""Evaluation for the SCoBots agent, adapted from agents/ppo/ppo_eval.py.

The eval env is built with `eval=True`: no reward clipping, no episodic life and
no concept reward, so the returns are the game's own score whatever REWARD_MODE
the agent was trained with.
"""

from typing import Callable

import flax
import jax
import jax.numpy as jnp

from jaxtari.environment import JaxEnvironment
from jaxtari.wrappers import JaxtariWrapper

from agents.run_utils import game_states


def load_model(model_path: str) -> tuple[dict, dict, dict]:
    """``(config, actor_params, critic_params)`` of a checkpoint written by scobots.py (used by viper.py)."""
    with open(model_path, "rb") as f:
        blob = flax.serialization.msgpack_restore(f.read())
    # to_bytes([config, [actor, critic]]) stores every sequence as {"0": ..., "1": ...}
    config, params = blob["0"], blob["1"]
    config["HIDDEN_DIMS"] = [config["HIDDEN_DIMS"][k] for k in sorted(config["HIDDEN_DIMS"], key=int)]
    return config, params["0"], params["1"]


def evaluate(
    params,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    Model: Callable,
    seed: int = 1,
):
    """Run `eval_episodes` episodes in parallel with the greedy policy.

    The greedy action is the argmax of the actor's logits. `params` are the
    SCoBots AgentParams (actor_params, critic_params); `Model(out_dim=n)` builds
    the actor. Returns the per-episode returns and the env states of the first
    episode up to its end (used for video logging).
    """
    env: JaxEnvironment | JaxtariWrapper = make_env(env_id)()
    actor = Model(out_dim=env.action_space().n)

    @jax.jit
    def wrapped_reset(key):
        """Reset the env and fix the observation shape to (1, n_concepts)."""
        obs, state = env.reset(key)
        return obs[None, ...], state

    @jax.jit
    def wrapped_step(state, action):
        """Step the env and fix the observation shape to (1, n_concepts)."""
        obs, state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return obs[None, ...], state, reward, done, info

    @jax.jit
    def greedy_action(obs):
        return jnp.argmax(actor.apply(params.actor_params, obs), axis=1)

    def step_fn(carry, _):
        obs, env_state = carry
        actions = jax.vmap(greedy_action)(obs)
        obs, env_state, reward, done, _ = jax.vmap(wrapped_step)(env_state, actions)
        first_state = jax.tree.map(lambda x: x[0], env_state)
        return (obs, env_state), (first_state, done, reward)

    @jax.jit
    def scanned_step(carry):
        return jax.lax.scan(step_fn, carry, None, length=1000)

    obs, env_states = jax.vmap(wrapped_reset)(jax.random.split(jax.random.PRNGKey(seed), eval_episodes))
    carry = (obs, env_states)

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
