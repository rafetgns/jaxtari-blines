"""
Reward machine for JAXAtari Frostbite.

The Frostbite-specific reward machine, event definitions, and task
decomposition were developed for this project.

The implementation follows the Reward Machine infrastructure provided by
this repository (GameRM, RewardMachine, and build_transitions). Game-specific
event detection is based on the object-centric Frostbite observations exposed
by JAXAtari.
"""


import functools

import jax
import jax.numpy as jnp

from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class FrostbiteRm(GameRM):
    # Events
    PROP_INDEX = {
        "igloo_progressed": 0,
        "igloo_regressed": 1,
        "igloo_completed": 2,
        "lost_life": 3,
        "reached_shore": 4,
        "closer_to_door": 5,
        "at_igloo_door": 6,
        "entered_igloo": 7,
    }

    TRANSITIONS = [
        # State 0: build the igloo
        {"from": 0, "true": ["lost_life"], "to": 0, "reward": -1.0},
        {"from": 0, "true": ["igloo_regressed"], "to": 0, "reward": -0.5},
        {"from": 0, "true": ["igloo_progressed"], "to": 0, "reward": 1.0},
        {"from": 0, "true": ["igloo_completed"], "to": 1, "reward": 5.0},

        # State 1: reach the shore
        {"from": 1, "true": ["lost_life"], "to": 1, "reward": -1.0},
        {"from": 1, "true": ["reached_shore"], "to": 2, "reward": 2.0},

        # State 2: move along the shore toward the real igloo door
        {"from": 2, "true": ["lost_life"], "to": 2, "reward": -1.0},
        {"from": 2, "true": ["at_igloo_door"], "to": 3, "reward": 5.0},
        {"from": 2, "true": ["closer_to_door"], "to": 2, "reward": 0.25},

        # State 3: enter the igloo
        {"from": 3, "true": ["lost_life"], "to": 3, "reward": -1.0},
        {"from": 3, "true": ["entered_igloo"], "to": 0, "reward": 20.0},
    ]

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX),
            self.PROP_INDEX,
            self.TRANSITIONS,
        )

    def num_states(self):
        return 4

    def init_state(self):
        return 0

    def terminal_state(self):
        return -99

    def from_states(self):
        return self._from

    def require_true(self):
        return self._rt

    def require_false(self):
        return self._rf

    def to_states(self):
        return self._to

    def rewards(self):
        return self._rew

    # igloo_progress counts placed blocks, 0..16; 16 = complete
    IGLOO_COMPLETE = 16

    # Frostbite constants in screen pixels
    SHORE_Y = 56.0
    SHORE_Y_TOLERANCE = 2.0
    TARGET_DOOR_X = 122.0  # FrostbiteConstants.TARGET_IGLOO_X
    DOOR_X_MIN, DOOR_X_MAX = 119.0, 126.0
    DOOR_Y_MIN, DOOR_Y_MAX = 50.0, 56.0

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, frames):
        bailey_x_now = frames.bailey.x[-1].astype(jnp.float32)
        bailey_y_now = frames.bailey.y[-1].astype(jnp.float32)
        bailey_active_now = frames.bailey.active[-1] > 0
        bailey_x_prev = frames.bailey.x[-2].astype(jnp.float32)
        bailey_y_prev = frames.bailey.y[-2].astype(jnp.float32)
        bailey_active_prev = frames.bailey.active[-2] > 0

        igloo_now, igloo_prev = frames.igloo_progress[-1], frames.igloo_progress[-2]
        lives_now, lives_prev = frames.lives[-1], frames.lives[-2]
        complete_now = igloo_now >= self.IGLOO_COMPLETE
        complete_prev = igloo_prev >= self.IGLOO_COMPLETE

        # -------------------------------------------------
        # Build skill
        # -------------------------------------------------

        igloo_completed = complete_now & ~complete_prev

        igloo_progressed = (
            (igloo_now > igloo_prev)
            & ~igloo_completed
        )

        igloo_regressed = (
            (igloo_now < igloo_prev)
            & bailey_active_now
        )

        lost_life = lives_now < lives_prev

        # -------------------------------------------------
        # Shore / navigation skill
        # -------------------------------------------------

        at_shore_now = jnp.abs(bailey_y_now - self.SHORE_Y) <= self.SHORE_Y_TOLERANCE
        at_shore_prev = jnp.abs(bailey_y_prev - self.SHORE_Y) <= self.SHORE_Y_TOLERANCE

        reached_shore = at_shore_now & bailey_active_now & complete_now

        at_igloo_door = (
                complete_now
                & bailey_active_now
                & (bailey_x_now >= self.DOOR_X_MIN)
                & (bailey_x_now <= self.DOOR_X_MAX)
                & (bailey_y_now >= self.DOOR_Y_MIN)
                & (bailey_y_now <= self.DOOR_Y_MAX)
        )

        # Reward movement that actually reduces horizontal distance
        # to the real door while Bailey is on the shore.
        distance_now = jnp.abs(bailey_x_now - self.TARGET_DOOR_X)
        distance_prev = jnp.abs(bailey_x_prev - self.TARGET_DOOR_X)

        closer_to_door = (
            at_shore_now
            & at_shore_prev
            & complete_now
            & bailey_active_now
            & (distance_now < distance_prev)
            & ~at_igloo_door
        )

        # -------------------------------------------------
        # Final entry skill
        # -------------------------------------------------

        entered_igloo = bailey_active_prev & ~bailey_active_now & complete_prev

        return jnp.array([
            igloo_progressed,
            igloo_regressed,
            igloo_completed,
            lost_life,
            reached_shore,
            closer_to_door,
            at_igloo_door,
            entered_igloo,
        ]).astype(jnp.int32)
