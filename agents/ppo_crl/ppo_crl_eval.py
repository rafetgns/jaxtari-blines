"""Evaluation for PPO-CRL.

Same protocol as agents/ppo/ppo_eval.py (greedy action = argmax of the actor's
logits, episodes run until done, the env built with eval=True, so rewards are not
clipped and a lost life does not end the episode). Continual learning evaluates
every task seen so far, many times per run, so the rollout is compiled once per
eval env with the parameters as an argument instead of baked in as constants.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from agents.run_utils import game_states


def make_evaluator(env, network, actor, eval_episodes: int, prepare_obs: Callable):
    """Returns evaluate(network_params, actor_params, seed) -> (episodic returns, env states of episode 0).

    prepare_obs maps one env's observation to the network's input shape (drops
    the pixel channel, zero-pads object-centric observations to a common size).
    """

    def reset_one(key):
        obs, state = env.reset(key)
        return prepare_obs(obs), state

    def step_one(state, action):
        obs, state, reward, terminated, truncated, info = env.step(state, action)
        return prepare_obs(obs), state, reward, jnp.logical_or(terminated, truncated)

    @jax.jit
    def reset(keys):
        return jax.vmap(reset_one)(keys)

    @jax.jit
    def scanned_step(network_params, actor_params, carry):
        def step_fn(carry, _):
            obs, env_state = carry
            logits = actor.apply(actor_params, network.apply(network_params, obs))
            actions = jnp.argmax(logits, axis=-1)
            obs, env_state, reward, done = jax.vmap(step_one)(env_state, actions)
            first_state = jax.tree.map(lambda x: x[0], env_state)
            return (obs, env_state), (first_state, done, reward)
        return jax.lax.scan(step_fn, carry, None, length=1000)

    def evaluate(network_params, actor_params, seed: int = 1):
        carry = reset(jax.random.split(jax.random.PRNGKey(seed), eval_episodes))

        first_states_chunks, done_chunks, reward_chunks = [], [], []
        done_ever = jnp.zeros(eval_episodes, dtype=jnp.bool_)
        while not jnp.all(done_ever):
            carry, (first_states, dones, rewards) = scanned_step(network_params, actor_params, carry)
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

        # the first episode's game states, for the eval video
        env_states_until_done = jax.tree.map(
            lambda x: x[: int(first_done[0]) + 1],
            game_states(first_states_history),
        )
        return episodic_returns, env_states_until_done

    return evaluate
