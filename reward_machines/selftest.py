"""Smoke-test reward machines on JAXtari with random play.

For every game: build the RM wrapper on the object-centric env, step a batch
of envs with uniformly random actions and report how often each proposition
is true and each transition fires, plus the RM reward and potential range.
A proposition that never fires under random play is not necessarily wrong
(some need skill), but one that is always true, or a crash, is.

    uv run python -m reward_machines.selftest            # all games
    uv run python -m reward_machines.selftest pong freeway --steps 2000
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import jaxtari
from jaxtari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
)

from reward_machines.reward_machine_wrapper import RewardMachineWrapper
from reward_machines.rm_registry import GAME_RM_REGISTRY


def make_rm_env(game: str, use_shaping: bool = True):
    env = AtariWrapper(jaxtari.make(game), sticky_actions=0.0, episodic_life=True,
                       first_fire=True, noop_max=30, full_action_space=False)
    env = FlattenObservationWrapper(NormalizeObservationWrapper(
        ObjectCentricWrapper(env, frame_stack_size=4, frame_skip=4, clip_reward=True)))
    return RewardMachineWrapper(env, GAME_RM_REGISTRY[game](), use_crm=True, use_shaping=use_shaping)


def run(game: str, n_envs: int, steps: int, seed: int = 0):
    env = make_rm_env(game)
    game_rm = env.rm
    names = sorted(GAME_RM_REGISTRY[game].PROP_INDEX, key=GAME_RM_REGISTRY[game].PROP_INDEX.get)
    n_actions = env.action_space().n

    obs, state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(seed), n_envs))
    step = jax.jit(jax.vmap(env.step))
    events = jax.jit(jax.vmap(lambda s: game_rm.get_events(env._frames(s.atari_state))))
    potential = jax.jit(jax.vmap(lambda s: game_rm.potential(env._frames(s.atari_state))))

    prop_counts = np.zeros(len(names))
    fired = np.zeros(len(game_rm.rewards), dtype=np.int64)
    rewards, phis, states_seen = [], [], set()
    for i in range(steps):
        actions = jax.random.randint(jax.random.PRNGKey(seed * 100_003 + i), (n_envs,), 0, n_actions)
        obs, state, reward, _, _, info = step(state, actions)
        prop_counts += np.asarray(events(state)).sum(axis=0)
        f = np.asarray(info["rm_fired_idx"])
        fired += np.bincount(f[f >= 0], minlength=len(fired))
        rewards.append(np.asarray(info["rm_reward"]))
        phis.append(np.asarray(potential(state)))
        states_seen.update(np.asarray(state.u).tolist())
    total = steps * n_envs
    rewards, phis = np.concatenate(rewards), np.concatenate(phis)
    assert obs.shape[-1] == env.observation_space().shape[0]
    assert np.isfinite(rewards).all() and np.isfinite(phis).all(), "non-finite RM reward or potential"
    return {
        "props": {n: prop_counts[k] / total for k, n in enumerate(names)},
        "fired": fired,
        "rm_reward": (rewards.min(), rewards.mean(), rewards.max()),
        "phi": (phis.min(), phis.max()),
        "states": sorted(states_seen),
        "num_states": game_rm.num_states,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("games", nargs="*", choices=sorted(GAME_RM_REGISTRY), metavar="GAME")
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    failures = 0
    for game in args.games or sorted(GAME_RM_REGISTRY):
        t0 = time.time()
        try:
            r = run(game, args.envs, args.steps)
        except Exception as exc:  # noqa: BLE001 - report every game
            failures += 1
            print(f"FAIL {game}: {type(exc).__name__}: {exc}")
            continue
        props = "  ".join(f"{n}={p:.4f}" for n, p in r["props"].items())
        lo, mean, hi = r["rm_reward"]
        print(f"ok   {game} ({time.time() - t0:.0f}s) | RM states visited {r['states']} of {r['num_states']} | "
              f"rm_reward [{lo:.2f}, {hi:.2f}] mean {mean:.4f} | phi [{r['phi'][0]:.2f}, {r['phi'][1]:.2f}]")
        print(f"     props (fraction of steps true): {props}")
        print(f"     transitions fired: {r['fired'].tolist()}")
    raise SystemExit(failures)


if __name__ == "__main__":
    main()
