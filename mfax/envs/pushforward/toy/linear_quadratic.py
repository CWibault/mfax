import jax
import jax.numpy as jnp
from flax import struct
from dataclasses import field

from mfax.envs.base.toy.linear_quadratic import (
    BaseLinearQuadraticEnvParams,
    BaseLinearQuadraticEnvironment,
    BaseLinearQuadraticAggregateState,
)
from mfax.envs.pushforward.base import (
    PushforwardEnvParams,
    PushforwardEnvironment,
    PushforwardAggregateState,
)


@struct.dataclass
class PushforwardLinearQuadraticAggregateState(
    PushforwardAggregateState, BaseLinearQuadraticAggregateState
):
    pass


@struct.dataclass
class PushforwardLinearQuadraticEnvParams(
    PushforwardEnvParams, BaseLinearQuadraticEnvParams
):
    # --- require default class so ordering stays valid under multiple inheritance ---
    states: jax.Array = field(default_factory=lambda: jnp.empty((0, 0)))


class PushforwardLinearQuadraticEnvironment(
    PushforwardEnvironment, BaseLinearQuadraticEnvironment
):
    @property
    def obs_dim(self) -> int:
        return 1

    def mf_reset_env(
        self, key: jax.Array
    ) -> tuple[jax.Array, PushforwardLinearQuadraticAggregateState]:

        # --- initial mean-field distribution ---
        mu_0 = jnp.ones(self.n_states) / self.n_states
        mu_mean = jnp.sum(mu_0 * self.params.discrete_states)

        # --- common noise ---
        z = jax.lax.select(
            self.params.common_noise,
            jax.lax.select(jax.random.bernoulli(key), 1, -1),
            0,
        )

        aggregate_s = PushforwardLinearQuadraticAggregateState(
            mu=mu_0, z=z, time=0, mu_mean=mu_mean
        )
        return self.get_shared_obs(aggregate_s), aggregate_s

    def mf_step_env(
        self,
        key: jax.Array,
        aggregate_s: PushforwardLinearQuadraticAggregateState,
        prob_a: jax.Array,
    ) -> tuple[
        jax.Array,
        PushforwardLinearQuadraticAggregateState,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        next_mu = self.mf_transition(aggregate_s.mu, prob_a, aggregate_s)
        next_mu_mean = jnp.sum(next_mu * self.params.discrete_states)
        next_time = aggregate_s.time + 1
        next_aggregate_s = PushforwardLinearQuadraticAggregateState(
            mu=next_mu, z=aggregate_s.z, time=next_time, mu_mean=next_mu_mean
        )

        terminated = self.is_terminal(next_time)
        truncated = self.is_truncated(next_time)

        mat_r_step, mat_r_term = self.mf_reward(aggregate_s, next_aggregate_s)
        mat_r = jax.lax.select(terminated, mat_r_term, mat_r_step)
        return (
            jax.lax.stop_gradient(self.get_shared_obs(next_aggregate_s)),
            jax.lax.stop_gradient(next_aggregate_s),
            jax.lax.stop_gradient(mat_r),
            jax.lax.stop_gradient(terminated),
            jax.lax.stop_gradient(truncated),
        )

    def _single_pushforward_step(
        self,
        state: int,
        action_idx: int,
        aggregate_s: PushforwardLinearQuadraticAggregateState,
    ):

        assert state.ndim == 0, "state must be an integer"
        assert action_idx.ndim == 0, f"action_idx ndim ({action_idx.ndim}) must be 0"

        action = self.params.actions[action_idx]

        # --- step single agent forward ---
        deterministic_next_state_idx = self._single_step(state, action, aggregate_s)

        idio_scale = self.params.sigma * jnp.sqrt(1.0 - (self.params.rho**2))
        idio_next_state_idxs = jnp.clip(
            jnp.round(
                deterministic_next_state_idx + idio_scale * self.params.idio_atoms
            ).astype(jnp.int32),
            0,
            self.params.num_states - 1,
        )

        next_state_idxs = jnp.concatenate(
            [idio_next_state_idxs, jnp.array([deterministic_next_state_idx])], axis=0
        )
        probs = jnp.concatenate(
            [
                self.params.idio_atoms_probs * self.params.idio_noise,
                jnp.array([1.0 - self.params.idio_noise]),
            ],
            axis=0,
        )
        probs = probs / jnp.where(probs.sum() > 0, probs.sum(), 1.0)
        return jax.lax.stop_gradient(next_state_idxs), jax.lax.stop_gradient(probs)

    def _single_pushforward_reward(
        self,
        state: int,
        action_idx: int,
        aggregate_s: PushforwardLinearQuadraticAggregateState,
        next_aggregate_s: PushforwardLinearQuadraticAggregateState,
    ):

        assert state.ndim == 0, "state must be an integer"
        assert action_idx.ndim == 0, f"action_idx ndim ({action_idx.ndim}) must be 0"

        action = self.params.actions[action_idx]

        # --- step single agent forward ---
        return self._single_reward(state, action, aggregate_s, next_aggregate_s)

    def get_shared_obs(
        self, aggregate_s: PushforwardLinearQuadraticAggregateState
    ) -> jax.Array:
        mu_mean = jnp.sum(aggregate_s.mu * self.params.discrete_states)
        return jnp.array(mu_mean).reshape(-1)

    def normalize_obs(
        self, individual_obs: jax.Array, normalize_obs: bool = False
    ) -> jax.Array:
        # --- normalize location of mean of mean-field ---
        normalized_obs = individual_obs / self.params.num_states
        return jax.lax.select(
            normalize_obs, normalized_obs, individual_obs.astype(jnp.float32)
        )

    def normalize_individual_s(
        self, individual_states: jax.Array, normalize_states: bool = False
    ) -> jax.Array:
        return individual_states
