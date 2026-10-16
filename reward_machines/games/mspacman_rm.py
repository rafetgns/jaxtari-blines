import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions

# pellets (252 grid slots) and power_pellets (4) are ObjectObservations; a
# slot is active while its pellet is uneaten (off-path slots are never active)
NUM_PELLETS = 252
NUM_POWER_PELLETS = 4


class MsPacmanRm(GameRM):
    """v4: power-pellet stage machine (same idea as Seaquest's diver count).

    u_k = number of power pellets eaten in the current level (k = 0..4).
    Power pellets are the strategic resource of a level (ghost modes are
    NOT in the observation), so the RM gives the agent an explicit stage:

        u_k --power_pellet_eaten--> u_{k+1}  +0.5  (option)
        u_k --pellet_eaten--------> u_k      +0.05 (option)
        u_k --level_cleared-------> u0       +5.0  (option, pellets refill)
        idle frames                          -0.01 (not an option)

    level_cleared also detects the pellet refill of a new level (the frame
    with 0 pellets left is not guaranteed to be observed).
    Shaping potential: fraction of pellets eaten in the current level.
    """

    PHI_SCALE = 1.0
    NUM_STAGES = NUM_POWER_PELLETS + 1

    PROP_INDEX = {"level_cleared": 0, "power_pellet_eaten": 1, "pellet_eaten": 2}

    TRANSITIONS = []
    for _k in range(NUM_POWER_PELLETS + 1):
        TRANSITIONS.append(
            {"from": _k, "true": ["level_cleared"], "to": 0, "reward": 5.0, "option": True})
        if _k < NUM_POWER_PELLETS:
            TRANSITIONS.append(
                {"from": _k, "true": ["power_pellet_eaten"], "false": ["level_cleared"],
                 "to": _k + 1, "reward": 0.5, "option": True})
        TRANSITIONS.append(
            {"from": _k, "true": ["pellet_eaten"], "false": ["power_pellet_eaten", "level_cleared"],
             "to": _k, "reward": 0.05, "option": True})
        TRANSITIONS.append(
            {"from": _k, "false": ["level_cleared", "power_pellet_eaten", "pellet_eaten"],
             "to": _k, "reward": -0.01})
    del _k

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX), self.PROP_INDEX, self.TRANSITIONS
        )

    def num_states(self):     return self.NUM_STAGES
    def init_state(self):     return 0
    def terminal_state(self): return -99
    def from_states(self):    return self._from
    def require_true(self):   return self._rt
    def require_false(self):  return self._rf
    def to_states(self):      return self._to
    def rewards(self):        return self._rew

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, frames):
        pellets_left = jnp.sum(frames.pellets.active, axis=-1)
        power_left = jnp.sum(frames.power_pellets.active, axis=-1)
        pellets_left_now, pellets_left_prev = pellets_left[-1], pellets_left[-2]

        pellet_eaten       = pellets_left_now < pellets_left_prev
        power_pellet_eaten = power_left[-1] < power_left[-2]
        # Pellets never increase inside a level, so an increase = next level.
        # (prev > 0 so a level whose empty frame WAS observed is not counted twice)
        level_cleared = (pellets_left_prev > 0) & (
            (pellets_left_now == 0) | (pellets_left_now > pellets_left_prev)
        )

        return jnp.array([level_cleared, power_pellet_eaten, pellet_eaten]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, frames):
        """Phi = PHI_SCALE * fraction of pellet slots already eaten in this level."""
        return self.PHI_SCALE * (1.0 - jnp.sum(frames.pellets.active[-1]) / NUM_PELLETS)
