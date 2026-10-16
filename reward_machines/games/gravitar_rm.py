import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions

# Inside a planet level the terrain ObjectObservation has active boxes (it has
# none on the map and in the saucer arena); every box carries the level's
# terrain sprite as visual_id, which tells levels apart.


class GravitarRm(GameRM):
    """v5: planet-visit machine (v4 + reward fixes, see TRANSITIONS).

    In JAXAtari a planet counts as cleared only when all enemies are gone AND
    the ship then flies out of the top of the level (jax_gravitar.py,
    `reset_level_win`). "Enemies already cleared, now leave" is not visible
    in the observation, so the RM tracks the phase:

        u0 = MAP       on the solar-system map
        u1 = IN_LEVEL  inside a planet level, enemies left
        u2 = CLEARED   all enemies of this level destroyed, must exit

        u0 --in_level---------> u1    0.0  (option: land on a planet)
        u1 --enemies_cleared--> u2   +2.0  (option)
        u1 --fuel / enemy-----> u1   +0.5 / +1.0 (options)
        u1 --!in_level--------> u0    0.0  (left without clearing)
        u2 --!in_level--------> u0   +3.0  (option: exit after clearing)
        u2 --fuel-------------> u2   +0.5  (option)
        any --lost_life-------> u0   -1.0

    `in_level` is a level (not edge) proposition, so the RM always re-syncs
    with where the ship actually is, even if a transition frame is missed.
    Kill / tank events only count inside the SAME level in both frames
    (objects vanish from the obs when leaving a level).
    """

    PHI_SCALE = 0.5

    PROP_INDEX = {
        "lost_life": 0,
        "fuel_tank_collected": 1,
        "enemy_destroyed": 2,
        "enemies_cleared": 3,
        "in_level": 4,
    }

    TRANSITIONS = [
        # No per-frame idle penalty (v4 had -0.01): with episodic_life a lost
        # life ends the episode, and -0.01/(1-gamma) = -1 equals the death
        # penalty, so surviving was worth no more than dying.
        # ---- u0: MAP ----
        {"from": 0, "true": ["lost_life"], "to": 0, "reward": -1.0},
        # 0 reward for landing: +0.2 could be farmed by entering and leaving.
        {"from": 0, "true": ["in_level"], "false": ["lost_life"], "to": 1, "reward": 0.0, "option": True},

        # ---- u1: IN_LEVEL ----
        {"from": 1, "true": ["lost_life"], "to": 0, "reward": -1.0},
        {"from": 1, "false": ["in_level", "lost_life"], "to": 0, "reward": 0.0},
        {"from": 1, "true": ["enemies_cleared"], "false": ["lost_life"], "to": 2, "reward": 2.0, "option": True},
        {"from": 1, "true": ["fuel_tank_collected"], "false": ["lost_life"], "to": 1, "reward": 0.5, "option": True},  # cleared row above wins ties
        {"from": 1, "true": ["enemy_destroyed"], "false": ["lost_life", "enemies_cleared", "fuel_tank_collected"], "to": 1, "reward": 1.0, "option": True},

        # ---- u2: CLEARED ----
        {"from": 2, "true": ["lost_life"], "to": 0, "reward": -1.0},
        {"from": 2, "false": ["in_level", "lost_life"], "to": 0, "reward": 3.0, "option": True},
        {"from": 2, "true": ["fuel_tank_collected"], "false": ["lost_life"], "to": 2, "reward": 0.5, "option": True},
    ]

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX), self.PROP_INDEX, self.TRANSITIONS
        )

    def num_states(self):     return 3
    def init_state(self):     return 0
    def terminal_state(self): return -99
    def from_states(self):    return self._from
    def require_true(self):   return self._rt
    def require_false(self):  return self._rf
    def to_states(self):      return self._to
    def rewards(self):        return self._rew

    @staticmethod
    def _in_level(frames):
        """(F,) bool: a planet level's terrain is on screen."""
        return jnp.any(frames.terrain.active > 0, axis=-1)

    @staticmethod
    def _level_id(frames):
        return jnp.max(frames.terrain.visual_id, axis=-1)

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, frames):
        tanks = jnp.sum(frames.fuel_tanks.active, axis=-1)
        enemies = jnp.sum(frames.enemies.active, axis=-1)
        in_level_all = self._in_level(frames)
        level_id = self._level_id(frames)

        in_level = in_level_all[-1]
        same_level = in_level & in_level_all[-2] & (level_id[-1] == level_id[-2])

        lost_life           = frames.lives[-1] < frames.lives[-2]
        fuel_tank_collected = same_level & (tanks[-1] < tanks[-2])
        enemy_destroyed     = same_level & (enemies[-1] < enemies[-2])
        enemies_cleared     = same_level & (enemies[-2] > 0) & (enemies[-1] == 0)

        return jnp.array([
            lost_life, fuel_tank_collected, enemy_destroyed, enemies_cleared, in_level,
        ]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, frames):
        """Inside a planet level: PHI_SCALE * mean(enemies gone, tanks gone). 0 on the map."""
        n_enemies = frames.enemies.active.shape[-1]
        n_tanks = frames.fuel_tanks.active.shape[-1]
        enemies_left = jnp.sum(frames.enemies.active[-1])
        tanks_left = jnp.sum(frames.fuel_tanks.active[-1])
        progress = 0.5 * ((1.0 - enemies_left / n_enemies) + (1.0 - tanks_left / n_tanks))
        return jnp.where(self._in_level(frames)[-1], self.PHI_SCALE * progress, 0.0)
