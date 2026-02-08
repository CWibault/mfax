import jax
import jax.numpy as jnp
from flax import struct

from mfax.envs.base.toy.linear_quadratic import (
    BaseLinearQuadraticEnvParams,
    BaseLinearQuadraticEnvironment,
    BaseLinearQuadraticAggregateState,
)
from mfax.envs.sample.base import (
    SampleEnvironment,
    SampleEnvParams,
    SampleIndividualState,
    SampleAggregateState,
)


@struct.dataclass
class SampleLinearQuadraticIndividualState(SampleIndividualState):
    pass


@struct.dataclass
class SampleLinearQuadraticAggregateState(
    SampleAggregateState, BaseLinearQuadraticAggregateState
):
    pass


@struct.dataclass
class SampleLinearQuadraticEnvParams(SampleEnvParams, BaseLinearQuadraticEnvParams):
    # number of agents representing mean field
    n_agents: int = 10000


class SampleLinearQuadraticEnvironment(
    SampleEnvironment, BaseLinearQuadraticEnvironment
):
    @property
    def obs_dim(self) -> int:
        return 1

    def mf_step_env(
        self,
        key: jax.Array,
        individual_s: SampleLinearQuadraticIndividualState,
        aggregate_s: SampleLinearQuadraticAggregateState,
        vec_a: jax.Array,
    ) -> tuple[
        jax.Array,
        SampleLinearQuadraticIndividualState,
        SampleLinearQuadraticAggregateState,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:

        step_rng = jax.random.split(key, self.params.n_agents)

        # --- step individual agents forward ---
        next_individual_s = jax.vmap(self._single_idio_step, in_axes=(0, 0, 0, None))(
            step_rng, individual_s, vec_a, aggregate_s
        )

        # --- update aggregate state using individual states ---
        next_mu = (
            jnp.bincount(next_individual_s.state, length=self.params.num_states)
            / self.params.n_agents
        )
        next_mu_mean = jnp.sum(next_mu * self.params.discrete_states)
        next_time = aggregate_s.time + 1
        next_aggregate_s = SampleLinearQuadraticAggregateState(
            # --- state does not include mean-field distribution, since not required in reward calculation ---
            z=aggregate_s.z,
            time=next_time,
            mu_mean=next_mu_mean,
        )

        # --- get observations of updated aggregate state for each individual agent ---
        next_individual_obs = jax.vmap(self.get_individual_obs, in_axes=(0, None))(
            next_individual_s, next_aggregate_s
        )

        # --- check for termination and truncation ---
        terminated = self.is_terminal(next_time)
        truncated = self.is_truncated(next_time)

        # --- select between step and terminated reward ---
        vec_r_term, vec_r_st = jax.vmap(
            self._single_idio_reward, in_axes=(0, 0, None, None)
        )(individual_s, vec_a, aggregate_s, next_aggregate_s)
        vec_r = jax.lax.select(terminated, vec_r_term, vec_r_st)
        return (
            jax.lax.stop_gradient(next_individual_obs),
            jax.lax.stop_gradient(next_individual_s),
            jax.lax.stop_gradient(next_aggregate_s),
            jax.lax.stop_gradient(vec_r),
            jax.lax.stop_gradient(terminated),
            jax.lax.stop_gradient(truncated),
        )

    def mf_reset_env(
        self, key: jax.Array
    ) -> tuple[
        jax.Array,
        SampleLinearQuadraticIndividualState,
        SampleLinearQuadraticAggregateState,
    ]:

        # --- reset rng ---
        reset_rng = jax.random.split(key, self.params.n_agents)

        # --- common noise ---
        z = jax.lax.select(
            self.params.common_noise,
            jax.lax.select(jax.random.bernoulli(key), 1, -1),
            0,
        )

        # --- dummy aggregate state ---
        dummy_mu_0 = jnp.ones(self.params.num_states) / self.params.num_states
        dummy_mu_mean = jnp.sum(dummy_mu_0 * self.params.discrete_states)
        dummy_aggregate_s = SampleLinearQuadraticAggregateState(
            z=z,
            time=0,
            mu_mean=dummy_mu_mean,
        )

        # --- sample individual states using dummy aggregate state ---
        individual_s = jax.vmap(self.sa_reset_env, in_axes=(0, None))(
            reset_rng, dummy_aggregate_s
        )

        # --- update aggregate state using individual states ---
        mu_0 = (
            jnp.bincount(individual_s.state, length=self.params.num_states)
            / self.params.n_agents
        )
        mu_mean = jnp.sum(mu_0 * self.params.discrete_states)
        aggregate_s = SampleLinearQuadraticAggregateState(
            # --- state does not include mean-field distribution, since not required in reward calculation ---
            z=z,
            time=0,
            mu_mean=mu_mean,
        )

        # --- get observations of updated aggregate state for each individual agent ---
        individual_obs = jax.vmap(self.get_individual_obs, in_axes=(0, None))(
            individual_s, aggregate_s
        )

        return individual_obs, individual_s, aggregate_s

    def _single_idio_step(
        self,
        key: jax.Array,
        individual_s: SampleLinearQuadraticIndividualState,
        action_idx: int,
        aggregate_s: SampleLinearQuadraticAggregateState,
    ) -> tuple[SampleLinearQuadraticIndividualState]:

        assert individual_s.state.ndim == 0, "individual_s must be an integer"
        assert action_idx.ndim == 0, f"action_idx ndim ({action_idx.ndim}) must be 0"

        action = self.params.actions[action_idx]

        # --- step single agent forward ---
        deterministic_next_state_idx = self._single_step(
            individual_s.state, action, aggregate_s
        )

        # --- idiosyncratic noise ---
        delta = jax.random.choice(
            key, self.params.idio_atoms, p=self.params.idio_atoms_probs
        )
        delta = delta * jnp.asarray(self.params.idio_noise, dtype=delta.dtype)
        idio_next_state_idx = jnp.clip(
            deterministic_next_state_idx + delta, 0, self.params.num_states - 1
        ).astype(jnp.int32)

        # --- return next individual state ---
        next_individual_s = SampleLinearQuadraticIndividualState(
            state=idio_next_state_idx, time=individual_s.time + 1
        )
        return next_individual_s

    def _single_idio_reward(
        self,
        individual_s: SampleLinearQuadraticIndividualState,
        action_idx: int,
        aggregate_s: SampleLinearQuadraticAggregateState,
        next_aggregate_s: SampleLinearQuadraticAggregateState,
    ) -> tuple[jax.Array, jax.Array]:

        assert individual_s.state.ndim == 0, "individual_s must be an integer"
        assert action_idx.ndim == 0, f"action_idx ndim ({action_idx.ndim}) must be 0"

        action = self.params.actions[action_idx]

        # --- calculate reward ---
        return self._single_reward(
            individual_s.state, action, aggregate_s, next_aggregate_s
        )

    def sa_step_env(
        self,
        key: jax.Array,
        individual_s: SampleLinearQuadraticIndividualState,
        action: int,
        aggregate_s: SampleLinearQuadraticAggregateState,
        next_aggregate_s: SampleLinearQuadraticAggregateState,
    ) -> tuple[SampleLinearQuadraticIndividualState, jax.Array, jax.Array]:

        # --- step single agent forward ---
        next_individual_s = self._single_idio_step(
            key, individual_s, action, aggregate_s
        )
        r_step, r_term = self._single_idio_reward(
            individual_s, action, aggregate_s, next_aggregate_s
        )
        return (
            jax.lax.stop_gradient(next_individual_s),
            jax.lax.stop_gradient(r_step),
            jax.lax.stop_gradient(r_term),
        )

    def sa_reset_env(
        self, key: jax.Array, aggregate_s: SampleLinearQuadraticAggregateState
    ) -> SampleLinearQuadraticIndividualState:

        # --- initial mean-field distribution ---
        mu_0 = jnp.ones(self.params.num_states) / self.params.num_states

        state = jax.random.choice(key, self.params.states, p=mu_0).squeeze()
        return SampleLinearQuadraticIndividualState(state=state, time=0)

    def get_individual_obs(
        self,
        individual_s: SampleLinearQuadraticIndividualState,
        aggregate_s: SampleLinearQuadraticAggregateState,
    ) -> jax.Array:
        return jnp.array(aggregate_s.mu_mean).reshape(-1)

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
