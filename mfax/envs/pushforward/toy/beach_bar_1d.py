import jax
import jax.numpy as jnp
from flax import struct
from dataclasses import field

from mfax.envs.base.toy.beach_bar_1d import (
    BaseBeachBar1DEnvParams,
    BaseBeachBar1DEnvironment,
    BaseBeachBar1DAggregateState,
)
from mfax.envs.pushforward.base import (
    PushforwardEnvParams,
    PushforwardEnvironment,
    PushforwardAggregateState,
)


@struct.dataclass
class PushforwardBeachBar1DAggregateState(
    PushforwardAggregateState, BaseBeachBar1DAggregateState
):
    mu_mean: jax.Array


@struct.dataclass
class PushforwardBeachBar1DEnvParams(PushforwardEnvParams, BaseBeachBar1DEnvParams):
    # --- require default class so ordering stays valid under multiple inheritance ---
    states: jax.Array = field(default_factory=lambda: jnp.empty((0, 0)))


class PushforwardBeachBar1DEnvironment(
    PushforwardEnvironment, BaseBeachBar1DEnvironment
):
    @property
    def obs_dim(self) -> int:
        return 3

    def mf_reset_env(
        self, key: jax.Array
    ) -> tuple[jax.Array, PushforwardBeachBar1DAggregateState]:

        # --- sample bar location at IQR of num_states ---
        bar_loc_min = jnp.clip(
            jnp.floor(0.25 * self.params.num_states), 0, self.params.num_states - 1
        )
        bar_loc_max = jnp.clip(
            jnp.ceil(0.75 * self.params.num_states),
            bar_loc_min + 1,
            self.params.num_states,
        )
        bar_loc = self.params.discrete_states[
            jax.random.randint(
                key,
                minval=bar_loc_min,
                maxval=bar_loc_max,
                shape=(),
            )
        ]

        # --- initial mean-field distribution ---
        mu_0 = jnp.ones(self.n_states) / self.n_states
        mu_0 = mu_0.at[bar_loc].set(0.0)
        mu_0 = mu_0 / jnp.sum(mu_0)
        mu_mean = jnp.sum(mu_0 * self.params.discrete_states)

        # --- common noise is whether bar is open ---
        z = jnp.array(1, dtype=jnp.int32)
        aggregate_s = PushforwardBeachBar1DAggregateState(
            mu=mu_0, z=z, time=0, bar_loc=bar_loc, mu_mean=mu_mean
        )
        return self.get_shared_obs(aggregate_s), aggregate_s

    def mf_step_env(
        self,
        key: jax.Array,
        aggregate_s: PushforwardBeachBar1DAggregateState,
        prob_a: jax.Array,
    ) -> tuple[
        jax.Array, PushforwardBeachBar1DAggregateState, jax.Array, jax.Array, jax.Array
    ]:
        # --- update aggregate state ---
        next_mu = self.mf_transition(aggregate_s.mu, prob_a, aggregate_s)
        next_mu_mean = jnp.sum(next_mu * self.params.discrete_states)
        next_time = aggregate_s.time + 1
        next_z = jax.lax.select(
            self.params.common_noise
            & (next_time == (self.params.max_steps_in_episode // 2)),
            jax.random.bernoulli(key).astype(jnp.int32),
            aggregate_s.z.astype(jnp.int32),
        )
        next_aggregate_s = PushforwardBeachBar1DAggregateState(
            mu=next_mu,
            z=next_z,
            time=next_time,
            bar_loc=aggregate_s.bar_loc,
            mu_mean=next_mu_mean,
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
        aggregate_s: PushforwardBeachBar1DAggregateState,
    ):

        assert state.ndim == 0, "state must be an integer"
        assert action_idx.ndim == 0, f"action_idx ndim ({action_idx.ndim}) must be 0"

        action = self.params.actions[action_idx]

        # --- step single agent forward ---
        deterministic_next_state_idx = self._single_step(state, action, aggregate_s)

        idio_next_state_idxs = jnp.clip(
            deterministic_next_state_idx + self.params.idio_atoms,
            0,
            self.params.num_states - 1,
        )
        idio_next_state_idxs = jax.vmap(
            self._project_to_legal, in_axes=(None, 0, None)
        )(state, idio_next_state_idxs, aggregate_s.bar_loc)

        next_state_idxs = jnp.concatenate(
            [idio_next_state_idxs, jnp.array([deterministic_next_state_idx])], axis=0
        ).astype(jnp.int32)
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
        aggregate_s: PushforwardBeachBar1DAggregateState,
        next_aggregate_s: PushforwardBeachBar1DAggregateState,
    ):

        assert state.ndim == 0, "state must be an integer"
        assert action_idx.ndim == 0, f"action_idx ndim ({action_idx.ndim}) must be 0"

        action = self.params.actions[action_idx]

        # --- step single agent forward ---
        return self._single_reward(state, action, aggregate_s, next_aggregate_s)

    def get_shared_obs(
        self, aggregate_s: PushforwardBeachBar1DAggregateState
    ) -> jax.Array:
        return jnp.array(
            [
                aggregate_s.mu_mean,
                aggregate_s.z,
                aggregate_s.bar_loc,
            ]
        )

    def normalize_obs(
        self, shared_obs: jax.Array, normalize_obs: bool = False
    ) -> jax.Array:

        # --- normalize location of mean of mean-field and bar location ---
        normalized_obs = shared_obs.at[..., 0].set(
            1 - (shared_obs[..., 0] / self.params.num_states)
        )
        normalized_obs = normalized_obs.at[..., 2].set(
            1 - (shared_obs[..., 2] / self.params.num_states)
        )
        return jax.lax.select(
            normalize_obs, normalized_obs, shared_obs.astype(jnp.float32)
        )

    def normalize_individual_s(
        self, individual_states: jax.Array, normalize_states: bool = False
    ) -> jax.Array:
        return individual_states
