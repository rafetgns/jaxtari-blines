import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions

# Positions are recovered in the env's own coordinates: the observation shifts
# objects by half their size (x - width // 2, y - height // 2).
GATE_HALF_WIDTH = 16.0   # consts.flag_distance = 32 px between the poles
SKIER_Y = 46.0           # consts.skier_y, the line flags must cross
ALIGN_RANGE = 80.0       # half the screen width (160 px)

# NOTE: despite its name, `successful_gates` is a COUNTDOWN of gates remaining
# (starts at 20, -1 per passed gate; env reward = prev - now). See
# jaxtari/games/jax_skiing.py. So a gate is passed when it DECREASES.


class SkiingRm(GameRM):
    """v5: gate-streak machine + time pressure.

    ALE Skiing punishes every missed gate at the end of the run, so passing
    gates CONSECUTIVELY is what matters. How many gates in a row were passed
    is history, not visible in the observation; the RM tracks it:

        u0 = NO_STREAK, u1 = 1 in a row, u2 = 2+ in a row

        u0 --gate_passed--> u1   +1.0  (option)
        u1 --gate_passed--> u2   +1.2  (option)
        u2 --gate_passed--> u2   +1.5  (option)
        any --gate_missed-> u0   -0.5
        every other step         -0.01 (time cost, not an option)

    gate_missed: a gate scrolled past the skier line (same test as the env's
    own `crossed`) without the gate counter going down in that frame.
    Shaping potential: alignment of the skier with the next gate's centre.
    """

    PHI_SCALE = 0.2
    PASS_REWARD = (1.0, 1.2, 1.5)
    MISS_REWARD = -0.5
    # Time pressure. The env (ALE reward) charges time on every frame and
    # -500 per missed gate at the end; without a time cost the RM variants had
    # no reason to go fast. Skiing has no lives, so a step cost cannot make
    # "dying early" attractive. 0.5 miss ~ 50 steps, close to the env ratio.
    STEP_REWARD = -0.01

    PROP_INDEX = {"gate_passed": 0, "gate_missed": 1}

    TRANSITIONS = [
        {"from": 0, "true": ["gate_passed"], "to": 1, "reward": PASS_REWARD[0], "option": True},
        {"from": 0, "true": ["gate_missed"], "false": ["gate_passed"], "to": 0, "reward": MISS_REWARD},
        {"from": 0, "false": ["gate_passed", "gate_missed"], "to": 0, "reward": STEP_REWARD},
        {"from": 1, "true": ["gate_passed"], "to": 2, "reward": PASS_REWARD[1], "option": True},
        {"from": 1, "true": ["gate_missed"], "false": ["gate_passed"], "to": 0, "reward": MISS_REWARD},
        {"from": 1, "false": ["gate_passed", "gate_missed"], "to": 1, "reward": STEP_REWARD},
        {"from": 2, "true": ["gate_passed"], "to": 2, "reward": PASS_REWARD[2], "option": True},
        {"from": 2, "true": ["gate_missed"], "false": ["gate_passed"], "to": 0, "reward": MISS_REWARD},
        {"from": 2, "false": ["gate_passed", "gate_missed"], "to": 2, "reward": STEP_REWARD},
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
    def _flag_y(flags):
        return flags.y + flags.height // 2

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, frames):
        gate_passed = frames.successful_gates[-1] < frames.successful_gates[-2]  # counts DOWN, see note above

        fy_now = self._flag_y(frames.flags)[-1]
        fy_prev = self._flag_y(frames.flags)[-2]
        crossed = jnp.any((fy_prev > SKIER_Y) & (fy_now <= SKIER_Y))  # same test as the env's `crossed`
        gate_missed = crossed & ~gate_passed

        return jnp.array([gate_passed, gate_missed]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, frames):
        """Phi = PHI_SCALE * (1 - |skier_x - centre of next gate| / 80 px); 0 if no gate ahead."""
        skier = jax.tree.map(lambda x: x[-1], frames.skier)
        flags = jax.tree.map(lambda x: x[-1], frames.flags)
        skier_x = skier.x + skier.width // 2
        fy = self._flag_y(flags)
        active = flags.active > 0

        ahead = active & (fy > SKIER_Y)                 # flags scroll up towards the skier
        next_idx = jnp.argmin(jnp.where(ahead, fy, jnp.inf))
        centre = flags.x[next_idx] + flags.width[next_idx] // 2 + GATE_HALF_WIDTH
        align = 1.0 - jnp.clip(jnp.abs(skier_x - centre) / ALIGN_RANGE, 0.0, 1.0)
        return jnp.where(jnp.any(ahead), self.PHI_SCALE * align, 0.0)
