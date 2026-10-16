from abc import ABC, abstractmethod
import functools

import jax
import jax.numpy as jnp


class GameRM(ABC):
    """Per-game reward machine: propositions, transitions and an optional potential.

    `get_events` and `potential` receive `frames`, the game's own observation
    (unnormalised, with named fields) stacked over the wrapper's frame stack:
    every leaf has a leading axis of length 4, oldest frame first, so
    `frames.player.y[-1]` is the current and `frames.player.y[-2]` the previous
    agent step's value. Reading named fields instead of positions in the
    flattened vector keeps the labelling valid when a game's observation grows.
    """

    TRANSITIONS = []

    @abstractmethod
    def num_states(self) -> int:
        """Number of RM states (including the terminal state)."""
        ...

    @abstractmethod
    def init_state(self) -> int:
        """Index of the initial RM state."""
        ...

    @abstractmethod
    def terminal_state(self) -> int:
        """Index of the single terminal state."""
        ...

    @abstractmethod
    def from_states(self) -> jnp.ndarray:
        """Array which saves starting state of Transistion i"""
        ...

    @abstractmethod
    def to_states(self) -> jnp.ndarray:
        """Array which saves destination state of Transistion i"""
        ...

    @abstractmethod
    def require_false(self) -> jnp.ndarray:
        """Saves which props need to be false in a Transistion"""
        ...

    @abstractmethod
    def require_true(self) -> jnp.ndarray:
        """Saves which props need to be true in a Transistion"""
        ...

    @abstractmethod
    def rewards(self) -> jnp.ndarray:
        """Saves reward of a Transistion"""
        ...

    @abstractmethod
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, frames) -> jnp.ndarray:
        """Game-specific labelling: stacked observation frames -> int32 proposition vector."""
        ...

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, frames) -> jnp.ndarray:
        """Shaping potential Phi(s). Default 0.0 means no shaping for this game."""
        return jnp.zeros(())
