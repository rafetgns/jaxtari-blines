import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions

# distances are measured in screen fractions, as in the normalised observation
SCREEN_W, SCREEN_H = 160.0, 210.0


class MontezumaRm(GameRM):
    """v4: key -> door machine (Icarte et al. style).

    The inventory (keys, sword, ...) is NOT part of the observation, but a
    door only opens when the player carries a key. The RM state is exactly
    that missing memory:

        u0 = EMPTY_HANDED   looking for an item (key)
        u1 = CARRYING       has picked something up, should use it on a door

        u0 --item_collected--> u1   +1.0  (option)
        u1 --door_opened-----> u0   +2.0  (option)
        u1 --item_collected--> u1   +0.5  (option, e.g. second key / sword)
        every other frame           +0.01 survival bonus (not an option)

    Events are only counted inside the same room: items/doors are reloaded
    per room, so a room change would otherwise look like a pick-up / opening.
    Shaping potential: closeness of the player to the nearest active item.
    """

    PHI_SCALE = 0.2

    PROP_INDEX = {"item_collected": 0, "door_opened": 1}

    TRANSITIONS = [
        # ---- u0: EMPTY_HANDED ----
        {"from": 0, "true": ["item_collected"], "to": 1, "reward": 1.0, "option": True},
        # door without a tracked item (e.g. a key picked up earlier): pay, stay
        {"from": 0, "true": ["door_opened"], "false": ["item_collected"], "to": 0, "reward": 2.0},
        {"from": 0, "false": ["item_collected", "door_opened"], "to": 0, "reward": 0.01},

        # ---- u1: CARRYING ----
        {"from": 1, "true": ["door_opened"], "to": 0, "reward": 2.0, "option": True},
        {"from": 1, "true": ["item_collected"], "to": 1, "reward": 0.5, "option": True},  # door row above wins ties
        {"from": 1, "false": ["item_collected", "door_opened"], "to": 1, "reward": 0.01},
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

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, frames):
        # static room geometry: the platforms' x, y and width change only with the room
        p = frames.platforms
        geom = jnp.concatenate([p.x, p.y, p.width], axis=-1)
        same_room = jnp.all(geom[-1] == geom[-2])

        items = jnp.sum(frames.items.active, axis=-1)
        doors = jnp.sum(frames.doors.active, axis=-1)

        item_collected = same_room & (items[-1] < items[-2])
        door_opened    = same_room & (doors[-1] < doors[-2])

        return jnp.array([item_collected, door_opened]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, frames):
        """Phi = PHI_SCALE * (1 - distance to nearest active item, in screen fractions); 0 if the room has none."""
        px, py = frames.player.x[-1], frames.player.y[-1]
        items = jax.tree.map(lambda x: x[-1], frames.items)
        active = items.active > 0

        dist = jnp.where(active, jnp.hypot((items.x - px) / SCREEN_W, (items.y - py) / SCREEN_H), jnp.inf)
        nearest = jnp.min(dist)
        return jnp.where(
            jnp.any(active),
            self.PHI_SCALE * (1.0 - jnp.clip(nearest, 0.0, 1.0)),
            0.0,
        )
