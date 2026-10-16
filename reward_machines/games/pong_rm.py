import functools
import numpy as np
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class PongRm(GameRM):
    PROP_INDEX = {"scored": 0, "conceded": 1}

    # transitions: one state (u0), two events. order = priority.
    TRANSITIONS = [
        {"from": 0, "true": ["scored"],   "to": 0, "reward":  1.0},
        {"from": 0, "true": ["conceded"], "to": 0, "reward": -1.0},
    ]

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX), self.PROP_INDEX, self.TRANSITIONS
        )

    def num_states(self):     return 1
    def init_state(self):     return 0
    def terminal_state(self): return -99

    def from_states(self):    return self._from
    def require_true(self):   return self._rt
    def require_false(self):  return self._rf
    def to_states(self):      return self._to
    def rewards(self):        return self._rew

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, frames):
        # score changes between the two most recent agent steps
        scored = frames.score_player[-1] > frames.score_player[-2]
        conceded = frames.score_enemy[-1] > frames.score_enemy[-2]
        return jnp.array([scored, conceded]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, frames):
        return jnp.zeros(())
