from typing import Any
from functools import partial
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import struct

from mfax.envs.base.base import BaseEnvironment, BaseAggregateState, BaseMFSequence


@struct.dataclass
class SampleIndividualState:
    # --- state ---
    state: jax.Array
    time: int = 0


@struct.dataclass
class SampleAggregateState(BaseAggregateState):
    pass


@struct.dataclass
class SampleMFSequence(BaseMFSequence):
    vec_a: jax.Array
    vec_r: jax.Array


@struct.dataclass
class SampleEnvParams:
    n_agents: int


class SampleEnvironment(BaseEnvironment, ABC):
    """
    Abstract base class for all Sample-based environments.
    """

    @partial(jax.jit, static_argnames=("self",))
    def mf_step(
        self,
        key: jax.Array,
        vec_individual_s: SampleIndividualState,
        aggregate_s: SampleAggregateState,
        vec_a: jax.Array,
    ) -> tuple[
        jax.Array,
        jax.Array,
        SampleIndividualState,
        SampleIndividualState,
        SampleAggregateState,
        SampleAggregateState,
        jax.Array,
        jax.Array,
        jax.Array,
        dict[Any, Any],
    ]:

        key_step, key_reset = jax.random.split(key)

        (
            vec_individual_obs_st,
            vec_individual_s_st,
            aggregate_s_st,
            vec_r,
            aggregate_terminated,
            aggregate_truncated,
        ) = self.mf_step_env(key_step, vec_individual_s, aggregate_s, vec_a)
        vec_individual_obs_re, vec_individual_s_re, aggregate_s_re = self.mf_reset_env(
            key_reset
        )

        # --- choose between reset and non-reset states and observations based on whether environment is terminated or truncated. ---
        aggregate_done = jnp.logical_or(aggregate_terminated, aggregate_truncated)
        aggregate_s = jax.tree.map(
            lambda x, y: jax.lax.select(aggregate_done, x, y),
            aggregate_s_re,
            aggregate_s_st,
        )
        vec_individual_s = jax.tree.map(
            lambda x, y: jax.lax.select(aggregate_done, x, y),
            vec_individual_s_re,
            vec_individual_s_st,
        )
        vec_individual_obs = jax.lax.select(
            aggregate_done, vec_individual_obs_re, vec_individual_obs_st
        )
        return (
            vec_individual_obs,
            vec_individual_obs_st,
            vec_individual_s,
            vec_individual_s_st,
            aggregate_s,
            aggregate_s_st,
            vec_r,
            aggregate_terminated,
            aggregate_truncated,
            {},
        )

    @partial(jax.jit, static_argnames=("self",))
    def mf_reset(
        self,
        key: jax.Array,
    ) -> tuple[jax.Array, SampleAggregateState]:
        vec_individual_obs, vec_individual_s, aggregate_s = self.mf_reset_env(key)
        return vec_individual_obs, vec_individual_s, aggregate_s

    @abstractmethod
    def mf_step_env(
        self,
        key: jax.Array,
        vec_individual_s: SampleIndividualState,
        aggregate_s: SampleAggregateState,
        vec_a: jax.Array,
    ) -> tuple[
        jax.Array,
        SampleIndividualState,
        SampleAggregateState,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        """
        Steps environment forward for a given aggregate state and vector of actions for each agent.
        """
        raise NotImplementedError

    @abstractmethod
    def mf_reset_env(
        self, key: jax.Array
    ) -> tuple[jax.Array, SampleIndividualState, SampleAggregateState]:
        """
        Resets Mean Field distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def _single_idio_step(
        self,
        key: jax.Array,
        individual_s: SampleIndividualState,
        action: jax.Array,
        aggregate_s: SampleAggregateState,
    ) -> tuple[SampleIndividualState]:
        """
        Returns the next individual state for a single agent with idiosyncratic noise (i.e. stochastic step forward).
        """
        raise NotImplementedError

    @abstractmethod
    def _single_idio_reward(
        self,
        individual_s: SampleIndividualState,
        action: jax.Array,
        aggregate_s: SampleAggregateState,
        next_aggregate_s: SampleAggregateState,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Returns reward for a current state, action and aggregate state.
        """
        raise NotImplementedError

    def sa_step(
        self,
        key: jax.Array,
        mf_sequence: SampleMFSequence,
        individual_s: SampleIndividualState,
        action: jax.Array,
    ) -> tuple[jax.Array, SampleIndividualState, jax.Array, jax.Array, jax.Array]:
        """
        Steps single agent forward for a given mean-field sequence and action.
        """
        key_step, key_reset = jax.random.split(key)

        # --- step and reset rng ---
        key_step, key_reset = jax.random.split(key)

        # --- identify relevant aggregate states from mean-field sequence ---
        aggregate_s = jax.tree_map(
            lambda x: jnp.take(x, individual_s.time, axis=0), mf_sequence.aggregate_s
        )
        next_aggregate_s = jax.tree_map(
            lambda x: jnp.take(x, individual_s.time + 1, axis=0),
            mf_sequence.aggregate_s,
        )
        reset_aggregate_s = jax.tree_map(
            lambda x: jnp.take(x, 0, axis=0), mf_sequence.aggregate_s
        )

        # --- step single agent forward ---
        individual_s_step, reward_step, reward_term = self.sa_step_env(
            key_step, individual_s, action, aggregate_s, next_aggregate_s
        )
        individual_obs_step = self.get_individual_obs(
            individual_s_step, next_aggregate_s
        )
        individual_s_reset = self.sa_reset_env(key_reset, reset_aggregate_s)
        individual_obs_reset = self.get_individual_obs(
            individual_s_reset, reset_aggregate_s
        )

        # --- observation, termination and truncation are based on next aggregate state ---
        aggregate_terminated = jax.tree_map(
            lambda x: jnp.take(x, individual_s.time + 1, axis=0),
            mf_sequence.aggregate_terminated,
        )
        aggregate_truncated = jax.tree_map(
            lambda x: jnp.take(x, individual_s.time + 1, axis=0),
            mf_sequence.aggregate_truncated,
        )

        # --- choose between reset and non-reset state based on whether environment is terminated or truncated ---
        aggregate_done = jnp.logical_or(aggregate_terminated, aggregate_truncated)
        individual_s = jax.tree.map(
            lambda x, y: jax.lax.select(aggregate_done, x, y),
            individual_s_reset,
            individual_s_step,
        )
        individual_obs = jax.lax.select(
            aggregate_done, individual_obs_reset, individual_obs_step
        )

        reward = jax.lax.select(aggregate_done, reward_term, reward_step)
        return (
            individual_obs,
            individual_s,
            reward,
            aggregate_terminated,
            aggregate_truncated,
        )

    def sa_reset(
        self, key: jax.Array, mf_sequence: SampleMFSequence
    ) -> tuple[jax.Array, SampleIndividualState]:
        """
        Resets single agent for a given mean-field sequence.
        """
        # --- identify relevant aggregate state from mean-field sequence ---
        reset_aggregate_s = jax.tree_map(
            lambda x: jnp.take(x, 0, axis=0), mf_sequence.aggregate_s
        )

        # --- reset single agent ---
        individual_s = self.sa_reset_env(key, reset_aggregate_s)
        individual_obs = self.get_individual_obs(individual_s, reset_aggregate_s)
        return individual_obs, individual_s

    def sa_step_env(
        self,
        key: jax.Array,
        individual_s: SampleIndividualState,
        action: jax.Array,
        aggregate_s: SampleAggregateState,
        next_aggregate_s: SampleAggregateState,
    ) -> tuple[SampleIndividualState, jax.Array, jax.Array]:
        """
        Steps single agent forward for a given aggregate state and action.
        """
        raise NotImplementedError

    def sa_reset_env(
        self, key: jax.Array, reset_aggregate_s: SampleAggregateState
    ) -> SampleIndividualState:
        """
        Resets single agent for a given aggregate state.
        """
        raise NotImplementedError

    @abstractmethod
    def get_individual_obs(
        self, individual_s: SampleIndividualState, aggregate_s: SampleAggregateState
    ) -> jax.Array:
        """
        Gets individual observation of aggregate state.
        """
        raise NotImplementedError

    def normalize_obs(
        self, aggregate_obs: jax.Array, normalize_obs: bool = False
    ) -> jax.Array:
        """
        Transforms aggregate observation for feeding into policy network. Works on batched observations.
        """
        raise NotImplementedError

    def normalize_individual_s(
        self, individual_s: jax.Array, normalize_states: bool = False
    ) -> jax.Array:
        """
        Transforms individual state for feeding into policy network. Works on batched observations.
        """
        raise NotImplementedError
