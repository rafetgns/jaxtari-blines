import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions

NUM_BLOCKS = 108         # 3 rows of ObjectObservations (blocks_bot/mid/top), 36 each
SCREEN_W = 160.0


class BreakoutRm(GameRM):
    """v4: serve / rally machine.

    After a lost life JAXAtari freezes the ball until FIRE is pressed
    (`game_started` is reset to 0), but that flag is NOT in the observation
    (ball.active is always 1). The RM state carries it:

        u0 = SERVE   ball waiting, agent has to launch it
        u1 = RALLY   ball in play

        u0 --ball_launched--> u1   +0.1  (option)
        u1 --brick_hit------> u1   +0.2  (option)
        u1 --wall_cleared---> u1   +5.0  (option)
        u1 --lost_life------> u0   -1.0

    wall_cleared also detects the brick refill (the frame with 0 bricks left
    is not guaranteed to be observed).
    Shaping potential: bricks cleared + paddle/ball horizontal alignment.
    """

    PHI_BRICKS = 1.0
    PHI_ALIGN = 0.1
    MOVE_EPS = 1e-6

    PROP_INDEX = {"wall_cleared": 0, "brick_hit": 1, "lost_life": 2, "ball_launched": 3}

    # First matching row wins, so the most specific condition comes first.
    TRANSITIONS = [
        # ---- u0: SERVE ----
        {"from": 0, "true": ["ball_launched"], "false": ["lost_life"], "to": 1, "reward": 0.1, "option": True},
        {"from": 0, "true": ["lost_life"], "to": 0, "reward": -1.0},
        {"from": 0, "true": ["brick_hit"], "to": 1, "reward": 0.2},   # launch frame missed: re-sync

        # ---- u1: RALLY ----
        {"from": 1, "true": ["wall_cleared"], "to": 1, "reward": 5.0, "option": True},
        {"from": 1, "true": ["brick_hit"], "false": ["wall_cleared"], "to": 1, "reward": 0.2, "option": True},
        {"from": 1, "true": ["lost_life"], "false": ["brick_hit", "wall_cleared"], "to": 0, "reward": -1.0},
    ]

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX), self.PROP_INDEX, self.TRANSITIONS
        )

    def num_states(self):     return 2
    def init_state(self):     return 0
    def terminal_state(self): return -99
    def from_states(self):    return self._from
    def require_true(self):   return self._rt
    def require_false(self):  return self._rf
    def to_states(self):      return self._to
    def rewards(self):        return self._rew

    @staticmethod
    def _bricks_left(frames):
        """Remaining bricks per frame, (F,)."""
        return sum(jnp.sum(rows.active, axis=-1) for rows in (frames.blocks_bot, frames.blocks_mid, frames.blocks_top))

    def _ball_moved(self, frames, t):
        """Did the ball move between frame t-1 and frame t (negative indices, -1 = newest)?"""
        dx = frames.ball.x[t] - frames.ball.x[t - 1]
        dy = frames.ball.y[t] - frames.ball.y[t - 1]
        return (jnp.abs(dx) + jnp.abs(dy)) > self.MOVE_EPS

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, frames):
        remaining = self._bricks_left(frames)
        remaining_now, remaining_prev = remaining[-1], remaining[-2]

        brick_hit     = remaining_now < remaining_prev
        # Bricks never reappear inside a wall, so an increase = new wall.
        # (prev > 0 so a wall whose empty frame WAS observed is not counted twice)
        wall_cleared  = (remaining_prev > 0) & ((remaining_now == 0) | (remaining_now > remaining_prev))
        lost_life     = frames.lives[-1] < frames.lives[-2]
        # Ball starts moving after being still (served).
        ball_launched = self._ball_moved(frames, -1) & ~self._ball_moved(frames, -2)

        return jnp.array([wall_cleared, brick_hit, lost_life, ball_launched]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, frames):
        """Phi = PHI_BRICKS * bricks cleared fraction
               + PHI_ALIGN * (1 - |paddle centre - ball centre| / screen width)."""
        cleared = 1.0 - self._bricks_left(frames)[-1] / NUM_BLOCKS

        paddle_c = frames.player.x[-1] + 0.5 * frames.player.width[-1]
        ball_c = frames.ball.x[-1] + 0.5 * frames.ball.width[-1]
        align = 1.0 - jnp.clip(jnp.abs(paddle_c - ball_c) / SCREEN_W, 0.0, 1.0)

        return self.PHI_BRICKS * cleared + self.PHI_ALIGN * align
