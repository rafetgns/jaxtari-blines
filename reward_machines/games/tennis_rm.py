"""
Reward machine for JAXAtari Tennis.

The RM tracks four phases:

    u0 = READY
         Waiting for the next rally.

    u1 = SERVE_READY
         Player-side serve geometry is in a hit-ready configuration.

    u2 = RALLY
         The ball is actively in play.

    u3 = POST_POINT
         A point has just been scored. Wait until stale stacked ball motion
         has disappeared before returning to READY.

For player serves, the rally-start shaping reward can be split into:

    READY -> SERVE_READY : +0.10
    SERVE_READY -> RALLY : +0.15

If the intermediate serve-ready phase is not observed,
READY -> RALLY receives +0.25 directly.
"""

import functools

import jax
import jax.numpy as jnp

from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class TennisRm(GameRM):
    # Proposition order must match get_events().
    PROP_INDEX = {
        "ball_in_play": 0,
        "serve_ready": 1,
        "player_point": 2,
        "enemy_point": 3,
        "player_game_progress": 4,
        "enemy_game_progress": 5,
    }

    # HRM options (edges marked "option": True). Options are keyed by guard
    # formula, so edges with the same guard share one option policy:
    #   serve_ready   (u0)         -> get into serve position
    #   ball_in_play  (u0, u1)     -> start the rally
    #   player_point  (u0, u1, u2) -> win the point (also covers game progress,
    #                                 since player_point is set then too)
    #   !ball_in_play (u3)         -> wait for the next rally
    # Penalty edges (enemy_point / enemy_game_progress) are deliberately not
    # options, see RewardMachineWrapper._option_signals.
    PHI_SCALE = 0.1
    PHI_MAX_DIST = 0.8  # ~max player-ball distance observed, in screen-normalised units (x / 160, y / 210)

    SCREEN_W, SCREEN_H = 160.0, 210.0
    SERVE_X_TOLERANCE = 0.08 * SCREEN_W  # px; 0.08 of the screen width, as in the PR
    PLAYER_HEIGHT = 23.0                 # px
    SERVE_Y_TOLERANCE = 3.0              # px, hit tolerance

    # RM states:
    #
    # u0 = READY
    # u1 = SERVE_READY
    # u2 = RALLY
    # u3 = POST_POINT
    #
    # A higher-level score increment can coincide with a point increment,
    # so these transitions are checked first.
    TRANSITIONS = [
        # ---- u0: READY -------------------------------------------------
        {"from": 0, "true": ["player_game_progress"],
         "to": 3, "reward": 3.0},

        {"from": 0, "true": ["enemy_game_progress"],
         "to": 3, "reward": -3.0},

        {"from": 0, "true": ["player_point"],
         "to": 3, "reward": 1.0, "option": True},

        {"from": 0, "true": ["enemy_point"],
         "to": 3, "reward": -1.0},

        # Direct rally start when the intermediate serve-ready phase
        # is not observed.
        {"from": 0, "true": ["ball_in_play"],
         "to": 2, "reward": 0.25, "option": True},

        # Player-side serve geometry is in a hit-ready configuration.
        {"from": 0, "true": ["serve_ready"],
         "to": 1, "reward": 0.10, "option": True},

        # ---- u1: SERVE_READY -------------------------------------------
        {"from": 1, "true": ["player_game_progress"],
         "to": 3, "reward": 3.0},

        {"from": 1, "true": ["enemy_game_progress"],
         "to": 3, "reward": -3.0},

        {"from": 1, "true": ["player_point"],
         "to": 3, "reward": 1.0, "option": True},

        {"from": 1, "true": ["enemy_point"],
         "to": 3, "reward": -1.0},

        # Complete the rally-start shaping reward.
        {"from": 1, "true": ["ball_in_play"],
         "to": 2, "reward": 0.15, "option": True},

        # Keep SERVE_READY sticky until the rally starts or a score event
        # occurs; returning to READY here would allow repeated +0.10 rewards.

        # ---- u2: RALLY -------------------------------------------------
        {"from": 2, "true": ["player_game_progress"],
         "to": 3, "reward": 3.0},

        {"from": 2, "true": ["enemy_game_progress"],
         "to": 3, "reward": -3.0},

        {"from": 2, "true": ["player_point"],
         "to": 3, "reward": 1.0, "option": True},

        {"from": 2, "true": ["enemy_point"],
         "to": 3, "reward": -1.0},

        # ---- u3: POST_POINT --------------------------------------------
        # Wait until stale movement from the previous rally disappears.
        {"from": 3, "false": ["ball_in_play"],
         "to": 0, "reward": 0.0, "option": True},
    ]

    def __init__(self):
        (
            self._from,
            self._rt,
            self._rf,
            self._to,
            self._rew,
        ) = build_transitions(
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

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, frames):
        f32 = lambda v: jnp.asarray(v, jnp.float32)
        ball_x, ball_y = f32(frames.ball.x), f32(frames.ball.y)
        player_x, player_y = f32(frames.player.x), f32(frames.player.y)

        # --------------------------------------------------------------
        # Ball-in-play detector: the ball moved over both of the last two steps
        # --------------------------------------------------------------
        moved = (ball_x[1:] != ball_x[:-1]) | (ball_y[1:] != ball_y[:-1])  # (3,)
        ball_in_play = moved[-2] & moved[-1]

        # --------------------------------------------------------------
        # Serve-preparation detector
        # --------------------------------------------------------------
        #
        # Horizontal distance alone can also match enemy-serve layouts,
        # so both horizontal and vertical alignment are required.
        serve_now = f32(frames.is_serving_state[-1]) > 0.5
        x_aligned = jnp.abs(player_x[-1] - ball_x[-1]) <= self.SERVE_X_TOLERANCE
        y_aligned = jnp.abs((player_y[-1] + self.PLAYER_HEIGHT) - ball_y[-1]) <= self.SERVE_Y_TOLERANCE
        serve_ready = serve_now & x_aligned & y_aligned

        # --------------------------------------------------------------
        # Score events
        # --------------------------------------------------------------
        def gained(field):
            v = f32(field)
            return v[-1] > v[-2]

        player_game_progress = gained(frames.player_sets)
        enemy_game_progress = gained(frames.enemy_sets)
        player_point = gained(frames.player_points) | player_game_progress
        enemy_point = gained(frames.enemy_points) | enemy_game_progress

        return jnp.array([
            ball_in_play,
            serve_ready,
            player_point,
            enemy_point,
            player_game_progress,
            enemy_game_progress,
        ]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, frames):
        """Shaping potential: closeness of the player to the ball (last frame).

        Phi in [0, PHI_SCALE]; kept small relative to the +-1 / +-3 RM rewards.
        """
        dx = (jnp.asarray(frames.player.x[-1], jnp.float32) - frames.ball.x[-1]) / self.SCREEN_W
        dy = (jnp.asarray(frames.player.y[-1], jnp.float32) - frames.ball.y[-1]) / self.SCREEN_H
        dist = jnp.hypot(dx, dy)
        return self.PHI_SCALE * (1.0 - jnp.clip(dist / self.PHI_MAX_DIST, 0.0, 1.0))
