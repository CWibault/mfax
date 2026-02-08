import jax
import jax.numpy as jnp
from flax import struct

from mfax.envs.base.toy.beach_bar_1d import (
    BaseBeachBar1DEnvParams,
    BaseBeachBar1DEnvironment,
    BaseBeachBar1DAggregateState,
)
from mfax.envs.sample.base import (
    SampleEnvironment,
    SampleEnvParams,
    SampleIndividualState,
    SampleAggregateState,
)


@struct.dataclass
class SampleBeachBar1DIndividualState(SampleIndividualState):
    pass


@struct.dataclass
class SampleBeachBar1DAggregateState(
    SampleAggregateState, BaseBeachBar1DAggregateState
):
    mu_mean: jax.Array


@struct.dataclass
class SampleBeachBar1DEnvParams(SampleEnvParams, BaseBeachBar1DEnvParams):
    # number of agents representing mean field
    n_agents: int = 10000


class SampleBeachBar1DEnvironment(SampleEnvironment, BaseBeachBar1DEnvironment):
    @property
    def obs_dim(self) -> int:
        return 3

    def mf_step_env(
        self,
        key: jax.Array,
        individual_s: SampleBeachBar1DIndividualState,
        aggregate_s: SampleBeachBar1DAggregateState,
        vec_a: jax.Array,
    ) -> tuple[
        jax.Array,
        SampleBeachBar1DIndividualState,
        SampleBeachBar1DAggregateState,
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
        next_z = jax.lax.select(
            self.params.common_noise
            & (next_time == (self.params.max_steps_in_episode // 2)),
            jax.random.bernoulli(key).astype(jnp.int32),
            aggregate_s.z.astype(jnp.int32),
        )
        next_aggregate_s = SampleBeachBar1DAggregateState(
            mu=next_mu,  # mu is required for reward function.
            z=next_z,
            time=next_time,
            bar_loc=aggregate_s.bar_loc,
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
        jax.Array, SampleBeachBar1DIndividualState, SampleBeachBar1DAggregateState
    ]:

        # --- reset rng ---
        reset_rng = jax.random.split(key, self.params.n_agents)

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

        # --- common noise is whether bar is open ---
        z = jnp.array(1, dtype=jnp.int32)

        # --- dummy aggregate state ---
        dummy_mu_0 = jnp.ones(self.n_states) / self.n_states
        dummy_mu_0 = dummy_mu_0.at[bar_loc].set(0.0)
        dummy_mu_0 = dummy_mu_0 / jnp.sum(dummy_mu_0)
        dummy_mu_mean = jnp.sum(dummy_mu_0 * self.params.discrete_states)
        dummy_aggregate_s = SampleBeachBar1DAggregateState(
            mu=dummy_mu_0,
            z=z,
            time=0,
            bar_loc=bar_loc,
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
        aggregate_s = SampleBeachBar1DAggregateState(
            mu=mu_0,
            z=z,
            time=0,
            bar_loc=bar_loc,
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
        individual_s: SampleBeachBar1DIndividualState,
        action_idx: int,
        aggregate_s: SampleBeachBar1DAggregateState,
    ) -> tuple[SampleBeachBar1DIndividualState]:

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
        idio_next_state_idx = self._project_to_legal(
            individual_s.state, idio_next_state_idx, aggregate_s.bar_loc
        )

        # --- return next individual state ---
        next_individual_s = SampleBeachBar1DIndividualState(
            state=idio_next_state_idx, time=individual_s.time + 1
        )
        return next_individual_s

    def _single_idio_reward(
        self,
        individual_s: SampleBeachBar1DIndividualState,
        action_idx: int,
        aggregate_s: SampleBeachBar1DAggregateState,
        next_aggregate_s: SampleBeachBar1DAggregateState,
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
        individual_s: SampleBeachBar1DIndividualState,
        action: int,
        aggregate_s: SampleBeachBar1DAggregateState,
        next_aggregate_s: SampleBeachBar1DAggregateState,
    ) -> tuple[SampleBeachBar1DIndividualState, jax.Array, jax.Array]:

        # --- step individual agent forward ---
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
        self, key: jax.Array, aggregate_s: SampleBeachBar1DAggregateState
    ) -> SampleBeachBar1DIndividualState:

        # --- initial mean-field distribution ---
        mu_0 = jnp.ones(self.params.num_states) / self.params.num_states
        mu_0 = mu_0.at[aggregate_s.bar_loc].set(0.0)
        mu_0 = mu_0 / jnp.sum(mu_0)

        state = jax.random.choice(key, self.params.states, p=mu_0).squeeze()
        return SampleBeachBar1DIndividualState(state=state, time=0)

    def get_individual_obs(
        self,
        individual_s: SampleBeachBar1DIndividualState,
        aggregate_s: SampleBeachBar1DAggregateState,
    ) -> jax.Array:
        return jnp.array(
            [
                aggregate_s.mu_mean,
                aggregate_s.z,
                aggregate_s.bar_loc,
            ]
        )

    def normalize_obs(
        self, aggregate_obs: jax.Array, normalize_obs: bool = False
    ) -> jax.Array:

        # --- normalize location of mean of mean-field and bar location ---
        normalized_obs = aggregate_obs.at[..., 0].set(
            1 - (aggregate_obs[..., 0] / self.params.num_states)
        )
        normalized_obs = normalized_obs.at[..., 2].set(
            1 - (aggregate_obs[..., 2] / self.params.num_states)
        )
        return jax.lax.select(
            normalize_obs, normalized_obs, aggregate_obs.astype(jnp.float32)
        )

    def normalize_individual_s(
        self, individual_states: jax.Array, normalize_states: bool = False
    ) -> jax.Array:

        return individual_states
