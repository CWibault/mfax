from typing import Any
from functools import partial
from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct

from mfax.envs.base.base import BaseEnvironment, BaseMFSequence, BaseAggregateState


@struct.dataclass
class PushforwardAggregateState(BaseAggregateState):
    mu: jax.Array
    # --- z and time inherited from BaseAggregateState ---


@struct.dataclass
class PushforwardEnvParams:
    states: jax.Array


@struct.dataclass
class PushforwardMFSequence(BaseMFSequence):
    aggregate_obs: jax.Array
    aggregate_hidden: Optional[jax.Array]
    prob_a: jax.Array
    mat_r: jax.Array


class PushforwardEnvironment(BaseEnvironment, ABC):
    """Abstract base class for all Pushforward environments."""

    @partial(jax.jit, static_argnames=("self",))
    def mf_step(
        self,
        key: jax.Array,
        aggregate_s: PushforwardAggregateState,
        prob_a: jax.Array,
    ) -> tuple[
        jax.Array,
        jax.Array,
        PushforwardAggregateState,
        PushforwardAggregateState,
        jax.Array,
        jax.Array,
        jax.Array,
        dict[Any, Any],
    ]:
        key_step, key_reset = jax.random.split(key)

        (
            aggregate_obs_st,
            aggregate_s_st,
            mat_r,
            aggregate_terminated,
            aggregate_truncated,
        ) = self.mf_step_env(key_step, aggregate_s, prob_a)
        aggregate_obs_re, aggregate_s_re = self.mf_reset_env(key_reset)

        # --- Choose between reset and non-reset state based on whether the environment is terminated or truncated. ---
        aggregate_done = jnp.logical_or(aggregate_terminated, aggregate_truncated)
        aggregate_s = jax.tree.map(
            lambda x, y: jax.lax.select(aggregate_done, x, y),
            aggregate_s_re,
            aggregate_s_st,
        )
        aggregate_obs = jax.lax.select(
            aggregate_done, aggregate_obs_re, aggregate_obs_st
        )
        return (
            aggregate_obs,
            aggregate_obs_st,
            aggregate_s,
            aggregate_s_st,
            mat_r,
            aggregate_terminated,
            aggregate_truncated,
            {},
        )

    @partial(jax.jit, static_argnames=("self",))
    def mf_reset(
        self,
        key: jax.Array,
    ) -> tuple[jax.Array, PushforwardAggregateState]:
        aggregate_obs, aggregate_s = self.mf_reset_env(key)
        return aggregate_obs, aggregate_s

    @partial(jax.jit, static_argnames=("self",))
    def mf_expected_value(
        self, vec: jax.Array, prob_a: jax.Array, aggregate_s: PushforwardAggregateState
    ) -> jax.Array:
        """
        Functional representation of pre-multiplying by A matrix (expected value of next state).
        Vmaps over states and actions.
        Args:
            vec: (n_states, 1) vector to be pre-multiplied by A matrix
            prob_a: (n_states, n_actions) probability of each action for each state
            aggregate_s: aggregate state
        Returns:
            expected_values: (n_states, 1) expected values
        """

        # --- vmap over states ---
        def single_state(i):
            return jax.vmap(self._single_pushforward_step, in_axes=(None, 0, None))(
                self.state_indices[i], jnp.arange(self.n_actions), aggregate_s
            )

        next_state_idxs, next_state_probs = jax.vmap(single_state, in_axes=(0))(
            jnp.arange(self.n_states)
        )
        expected_values = jnp.sum(
            vec[next_state_idxs] * next_state_probs * prob_a[..., None], axis=(1, 2)
        )

        # --- no stop gradient ---
        return expected_values

    @partial(jax.jit, static_argnames=("self",))
    def mf_transition(
        self, mu: jax.Array, prob_a: jax.Array, aggregate_s: PushforwardAggregateState
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """
        Functional representation of pre-multiplying by transpose of A matrix (mean-field update).
        Vmaps over states and actions.
        Args:
            mu: (n_states, 1) current mean-field vector
            prob_a: (n_states, n_actions) probability of each action for each state
            aggregate_s: aggregate state
        Returns:
            next_mu: (n_states, 1) next mean-field vector
        """

        # --- vmap over states ---
        def single_state(i):
            # --- vmap over actions ---
            return jax.vmap(self._single_pushforward_step, in_axes=(None, 0, None))(
                self.state_indices[i], jnp.arange(self.n_actions), aggregate_s
            )

        next_state_idxs, next_state_probs = jax.vmap(single_state, in_axes=(0))(
            jnp.arange(self.n_states)
        )
        next_m = (
            jnp.zeros((self.n_states,))
            .at[next_state_idxs.reshape(-1)]
            .add(
                (mu[..., None, None] * next_state_probs * prob_a[..., None]).reshape(-1)
            )
        )
        return next_m

    @partial(jax.jit, static_argnames=("self",))
    def mf_reward(
        self,
        aggregate_s: PushforwardAggregateState,
        next_aggregate_s: PushforwardAggregateState,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """
        Caculates the reward for taking each action in each state.
        Vmaps over states and actions.
        Args:
            aggregate_s: current aggregate state
            next_aggregate_s: next aggregate state
        Returns:
            mat_r_step: (n_states, n_actions) reward for taking each action in each state
            mat_r_term: (n_states, n_actions) terminal reward for taking each action in each state
        """

        # --- vmap over states ---
        def single_state(i):
            # --- vmap over actions ---
            return jax.vmap(
                self._single_pushforward_reward, in_axes=(None, 0, None, None)
            )(
                self.state_indices[i],
                jnp.arange(self.n_actions),
                aggregate_s,
                next_aggregate_s,
            )

        mat_r_step, mat_r_term = jax.vmap(single_state, in_axes=(0))(
            jnp.arange(self.n_states)
        )
        return mat_r_step, mat_r_term

    @abstractmethod
    def mf_step_env(
        self,
        key: jax.Array,
        aggregate_s: PushforwardAggregateState,
        prob_a: jax.Array,
    ) -> tuple[jax.Array, PushforwardAggregateState, jax.Array, jax.Array, jax.Array]:
        raise NotImplementedError

    @abstractmethod
    def mf_reset_env(
        self, key: jax.Array
    ) -> tuple[jax.Array, PushforwardAggregateState]:
        """Resets Mean Field distribution."""
        raise NotImplementedError

    @abstractmethod
    def _single_pushforward_step(
        self, state_idx: int, action: int, aggregate_s: PushforwardAggregateState
    ) -> tuple[jax.Array, jax.Array]:
        """
        Returns the next indices and probabilities of the next state for a current state, action and aggregate state.
        """
        raise NotImplementedError

    @abstractmethod
    def _single_pushforward_reward(
        self,
        state_idx: int,
        action: int,
        aggregate_s: PushforwardAggregateState,
        next_aggregate_s: PushforwardAggregateState,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Calculates the (expected, if depends on next state) reward for a single pushforward step.
        """
        raise NotImplementedError

    @abstractmethod
    def get_shared_obs(self, aggregate_s: PushforwardAggregateState) -> jax.Array:
        """
        Gets shared observation of aggregate state (basis assumption for RSPG).
        """
        raise NotImplementedError

    def normalize_obs(
        self, shared_obs: jax.Array, normalize_obs: bool = False
    ) -> jax.Array:
        """
        Transform aggregate observation for feeding into policy network. Must work on batched observations.
        """
        raise NotImplementedError

    def normalize_individual_s(
        self, individual_s: jax.Array, normalize_states: bool = False
    ) -> jax.Array:
        """
        Transform individual state for feeding into policy network. Must work on batched observations.
        """
        raise NotImplementedError
