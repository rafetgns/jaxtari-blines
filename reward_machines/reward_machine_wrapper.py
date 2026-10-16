import functools
from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax import flatten_util

import jaxtari.spaces as spaces
from jaxtari.wrappers import (
    FlattenObservationWrapper,
    JaxtariWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
)

from reward_machines.games.game_rm import GameRM
from reward_machines.options import OptionSpec
from reward_machines.reward_machine import RewardMachine


@chex.dataclass(frozen=True)
class TimeStep:
    """One (counterfactual) transition, as stored by agents/rm_dqn."""
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    done: chex.Array


@struct.dataclass
class RewardMachineState:
    atari_state: Any        # the wrapped ObjectCentricState; the name lets state walkers reach the game state
    u: jnp.ndarray          # current RM state
    prev_obs: jnp.ndarray   # previous (flattened, normalised) observation, for CRM experiences
    prev_phi: jnp.ndarray   # shaping potential of the previous step


class RewardMachineWrapper(JaxtariWrapper):
    """Steps a reward machine alongside an object-centric env.

    Apply on top of FlattenObservationWrapper(NormalizeObservationWrapper(
    ObjectCentricWrapper(...))). Events are computed from the unnormalised
    frame stack of the ObjectCentricWrapper, unflattened back into the game's
    observation (see GameRM). The observation gets a one-hot of the RM state
    appended. During training the reward is the RM reward (plus the optional
    potential-based shaping); with `eval=True` it is the game's own reward, so
    evaluation scores the game whatever the RM pays.
    """

    def __init__(
        self,
        env: FlattenObservationWrapper,
        game_rm: GameRM,
        eval: bool = False,
        use_crm: bool = False,
        use_shaping: bool = False,
        gamma: float = 0.99,
        options: Optional[OptionSpec] = None,
    ):
        super().__init__(env)
        self.rm = RewardMachine(game_rm)
        self.eval = eval
        self.use_crm = use_crm
        self.use_shaping = use_shaping
        self.gamma = gamma
        self.options = options
        self.states = jnp.arange(self.rm.num_states)

        # Flatten / Normalize pass the state through, so the inner state is the
        # ObjectCentricState and its obs_stack holds the raw flattened frames.
        inner = env
        while not isinstance(inner, ObjectCentricWrapper):
            assert isinstance(inner, (FlattenObservationWrapper, NormalizeObservationWrapper)), (
                f"RewardMachineWrapper expects Flatten/Normalize(ObjectCentricWrapper), found {type(inner).__name__}"
            )
            inner = inner._env
        probe_obs, _ = inner._env.reset(jax.random.PRNGKey(0))
        # ObjectCentricWrapper flattens each frame with ravel_pytree (then casts to float32),
        # so its unravel inverts it; it expects the ravelled dtype (int32 for all-int observations)
        flat_probe, self._unravel = flatten_util.ravel_pytree(probe_obs)
        self._flat_dtype = flat_probe.dtype

        base = self._env.observation_space()
        self._observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(base.shape[0] + self.rm.num_states,),
            dtype=jnp.float32,
        )

    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def _augment_obs(self, obs, u):
        """Append the RM state, one-hot encoded, to the flattened observation."""
        return jnp.concatenate([obs, jax.nn.one_hot(u, self.rm.num_states)], axis=-1)

    def _frames(self, oc_state):
        """(F, D) raw flattened frame stack -> the game's observation with a leading F axis."""
        return jax.vmap(self._unravel)(oc_state.obs_stack.astype(self._flat_dtype))

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        obs, inner_state = self._env.reset(key)
        phi = self.rm.potential(self._frames(inner_state)) if self.use_shaping else jnp.zeros(())
        state = RewardMachineState(
            atari_state=inner_state,
            u=jnp.array(self.rm.init_state, dtype=jnp.int32),
            prev_obs=obs,
            prev_phi=phi,
        )
        return self._augment_obs(obs, state.u), state

    def _get_crm_experience(self, u, action, prev_obs, obs, true_props, shaping, env_done):
        # Events are computed once by the caller and shared across all RM states.
        next_u, rm_reward, _fired_idx, rm_done = self.rm.step_from_props(u, true_props)
        return TimeStep(
            obs=self._augment_obs(prev_obs, u),
            action=action,
            reward=rm_reward + shaping,
            next_obs=self._augment_obs(obs, next_u),
            done=jnp.logical_or(env_done, rm_done),
        )

    def _option_signals(self, u, true_props):
        clause = self.rm.clause_matches(true_props)  # (T,) bool
        option_rewards = jnp.max(
            self.options.scatter * clause[:, None].astype(jnp.float32), axis=0
        )  # (num_options,)
        # Only subgoal edges terminate an option. Including penalty edges here
        # would end every option after one step (at_surface_idle fires on
        # nearly every frame) and collapse HRM into flat DQN.
        out_of_u = self.options.is_option_edge & (self.rm.from_states == u)
        return option_rewards, jnp.any(clause & out_of_u)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        obs, inner_state, env_reward, terminated, truncated, info = self._env.step(state.atari_state, action)
        done = terminated | truncated
        episode_over = jnp.logical_or(info["env_done"], truncated)
        frames = self._frames(inner_state)
        # after an autoreset the stack holds the new episode's first frame: no events
        true_props = jnp.where(episode_over, 0, self.rm.get_events(frames))

        # Potential-based shaping with Phi(terminal) = 0
        if self.use_shaping:
            phi = self.rm.potential(frames)
            shaping = self.gamma * jnp.where(done, 0.0, phi) - state.prev_phi
        else:
            phi = shaping = jnp.zeros(())

        next_u, rm_reward, fired_idx, rm_done = self.rm.step_from_props(state.u, true_props)

        if self.use_crm:
            # one experience per RM state from the same env transition
            info["experiences"] = jax.vmap(
                self._get_crm_experience, in_axes=(0, None, None, None, None, None, None)
            )(self.states, action, state.prev_obs, obs, true_props, shaping, done)

        if self.options is not None:
            info["option_rewards"], info["option_terminate"] = self._option_signals(state.u, true_props)

        next_u = jnp.where(episode_over, self.rm.init_state, next_u)
        new_state = RewardMachineState(atari_state=inner_state, u=next_u, prev_obs=obs, prev_phi=phi)

        info["rm_fired_idx"] = fired_idx
        info["rm_reward"] = rm_reward
        info["shaping"] = shaping
        reward = env_reward if self.eval else rm_reward + shaping
        return self._augment_obs(obs, next_u), new_state, reward, jnp.logical_or(terminated, rm_done), truncated, info
