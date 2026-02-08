from typing import TypeVar, Generic
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import struct

TAggregateState = TypeVar("TAggregateState", bound="BaseAggregateState")
TEnvParams = TypeVar("TEnvParams", bound="BaseEnvParams")


@struct.dataclass
class BaseAggregateState:
    # --- common noise, time ---
    # mu not included, since if the state-space is too large, the mean-field vector mu will be too large to store in memory.
    # pushforward environments store mu, but sample environments not necessarily.
    z: jax.Array
    time: int


@struct.dataclass
class BaseMFSequence:
    aggregate_s: BaseAggregateState
    aggregate_terminated: jax.Array
    aggregate_truncated: jax.Array


@struct.dataclass
class BaseEnvParams:
    max_steps_in_episode: int
    idio_noise: bool
    common_noise: bool


class BaseEnvironment(Generic[TAggregateState, TEnvParams], ABC):
    """Abstract base class for all Model-Based Mean Field environments."""

    def __init__(self, params: BaseEnvParams):
        self.params = params

    @property
    def is_partially_observable(self) -> bool:
        """
        Whether environment is partially observable.
        """
        raise NotImplementedError

    @property
    def n_states(self) -> int:
        return self.params.states.shape[0]

    @property
    def state_indices(self) -> jax.Array:
        return jnp.arange(self.n_states)

    def _single_step(
        self, state: jax.Array, action: jax.Array, aggregate_s: BaseAggregateState
    ) -> tuple[jax.Array, jax.Array]:
        """
        Returns the next individual state for a single agent with no idiosyncratic noise (i.e. deterministic step forward).
        """
        raise NotImplementedError

    def _single_reward(
        self,
        state: jax.Array,
        action: jax.Array,
        aggregate_s: BaseAggregateState,
        next_aggregate_s: BaseAggregateState,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Calculates the (expected, if depends on next state) reward for a state, action and aggregate state. Returns the reward for a step and the terminal reward.
        """
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, time: int) -> jax.Array:
        """Checks whether Mean Field is terminal (for finite horizon environments)."""
        raise NotImplementedError

    @abstractmethod
    def is_truncated(self, time: int) -> jax.Array:
        """Checks whether Mean Field is truncated (for infinite horizon environments)."""
        raise NotImplementedError

    def discount(self, aggregate_s: BaseAggregateState) -> jax.Array:
        """Return zero discount if episode has terminated."""
        return jax.lax.select(self.is_terminal(aggregate_s.time), 0.0, 1.0)
