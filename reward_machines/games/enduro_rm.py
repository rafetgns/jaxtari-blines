import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class EnduroRm(GameRM):
    # Opponent window layout (EnduroConstants.opponent_slot_ys): base y and y-range of each slot
    SLOT_BASE_Y = (135.0, 115.0, 95.0, 85.0, 75.0, 69.0, 64.0)
    SLOT_HEIGHT = (20.0, 20.0, 20.0, 10.0, 10.0, 6.0, 5.0)

    PROP_INDEX = {"overtook": 0, "got_overtaken": 1}

    # Single state: mirrors the game's scoring (+1 per overtake, -1 per being overtaken)
    TRANSITIONS = [
        {"from": 0, "true": ["overtook"], "to": 0, "reward": 1.0},
        {"from": 0, "true": ["got_overtaken"], "to": 0, "reward": -1.0},
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

    def _window_phase(self, enemies):
        """Fractional position of the opponent window in [0, 1), or -1 if no car is visible.

        `enemies` is one frame's opponent group: one row per window slot, nearest
        first. All visible cars share the same fractional window index:
        y = base_y[slot] + floor(phase * slot_height[slot]).
        """
        present = enemies.active > 0  # empty slots, and cars hidden by fog or the bottom cutoff, are inactive
        phase = jnp.clip(
            (enemies.y.astype(jnp.float32) - jnp.array(self.SLOT_BASE_Y)) / jnp.array(self.SLOT_HEIGHT), 0.0, 0.999
        )
        n_visible = present.sum()
        mean_phase = jnp.sum(phase * present) / jnp.maximum(n_visible, 1)
        return jnp.where(n_visible > 0, mean_phase, -1.0)

    @staticmethod
    def _step_candidates(phase_a, phase_b):
        """Phase change a -> b without, with a forward and with a backward window wrap."""
        d = phase_b - phase_a
        return jnp.stack([d, d - 1.0, d + 1.0])

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, frames):
        # The game counts an overtake when the window advances by a slot while slot 0
        # (the nearest car) is occupied, and the reverse when it falls back.
        phase = jax.vmap(self._window_phase)(frames.enemies)  # (frames,)
        known = phase >= 0

        # At full speed the window moves up to ~0.6 slots per agent step, so a plain
        # |change| > 0.5 test confuses fast forward motion with a backward wrap. Pick
        # the wrap hypothesis closest to the previous step's motion instead.
        older = self._step_candidates(phase[-3], phase[-2])
        velocity = jnp.where(known[-3] & known[-2], older[jnp.argmin(jnp.abs(older))], 0.0)
        candidates = self._step_candidates(phase[-2], phase[-1])
        change = candidates[jnp.argmin(jnp.abs(candidates - velocity))]
        both_known = known[-2] & known[-1]
        window_forward = both_known & (phase[-2] + change >= 1.0)
        window_backward = both_known & (phase[-2] + change < 0.0)

        # the nearest car is hidden by the bottom cutoff just before it is passed,
        # so it counts if it was visible in any of the previous frames
        nearest_recent = jnp.any(frames.enemies.active[:-1, 0] > 0)
        nearest_now = frames.enemies.active[-1, 0] > 0

        overtook = window_forward & nearest_recent
        got_overtaken = window_backward & nearest_now
        return jnp.array([overtook, got_overtaken]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, frames):
        return jnp.zeros(())
