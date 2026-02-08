import jax
import jax.numpy as jnp
from flax import struct

from mfax.envs.base.macro.endogenous import (
    BaseEndogenousEnvironment,
    BaseEndogenousEnvParams,
    BaseEndogenousAggregateState,
)
from mfax.envs.sample.base import (
    SampleEnvironment,
    SampleEnvParams,
    SampleIndividualState,
    SampleAggregateState,
)


@struct.dataclass
class SampleEndogenousIndividualState(SampleIndividualState):
    pass


@struct.dataclass
class SampleEndogenousAggregateState(
    SampleAggregateState, BaseEndogenousAggregateState
):
    pass


@struct.dataclass
class SampleEndogenousEnvParams(SampleEnvParams, BaseEndogenousEnvParams):
    # number of agents representing mean field
    n_agents: int = 10000

    # states per dimension
    lower_bound: jax.Array = jnp.array([0.0, 0.1])
    upper_bound: jax.Array = jnp.array([99, 2.0])
    num_states: tuple[int, int] = (200, 5)

    # idiosyncratic noise parameters
    idio_atoms: jax.Array = jnp.array([-1, 0, 1])
    idio_atoms_probs: jax.Array = jnp.array([0.1, 0.8, 0.1])

    def __post_init__(self):
        BaseEndogenousEnvParams.__post_init__(self)
        pivots = jnp.maximum(jnp.abs(self.lower_bound), 0.25)
        discrete_states = [
            jnp.clip(jnp.geomspace(lb + p, ub + p, ns) - p, a_min=0)
            for lb, ub, ns, p in zip(
                self.lower_bound, self.upper_bound, self.num_states, pivots
            )
        ]
        states = jnp.stack(
            [sms.ravel() for sms in jnp.meshgrid(*discrete_states, indexing="ij")],
            axis=1,
        )
        object.__setattr__(self, "pivots", pivots)
        object.__setattr__(self, "discrete_states", discrete_states)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "n_states", len(states))


class SampleEndogenousEnvironment(SampleEnvironment, BaseEndogenousEnvironment):
    def _prices(
        self, individual_s: SampleEndogenousIndividualState, z: float
    ) -> tuple[float, float]:
        assert individual_s.state.ndim == 2, (
            "Individual state must be 2D array of shape (num_agents, num_state_dims)"
        )
        # --- sum over nuisance variable to obtain marginal distributions ---
        av_wealth = jnp.mean(individual_s.state[:, 0])
        av_income = jnp.mean(individual_s.state[:, 1])

        # --- prices ---
        interest_rate = (
            jnp.exp(z)
            * self.params.cobb_douglas_alpha
            * ((av_income / (av_wealth + 1e-6)) ** (1 - self.params.cobb_douglas_alpha))
        )
        wage = (
            jnp.exp(z)
            * (1 - self.params.cobb_douglas_alpha)
            * ((av_wealth / (av_income + 1e-6)) ** self.params.cobb_douglas_alpha)
        )
        return interest_rate, wage

    def mf_step_env(
        self,
        key: jax.Array,
        individual_s: SampleEndogenousIndividualState,
        aggregate_s: SampleEndogenousAggregateState,
        vec_a: jax.Array,
    ) -> tuple[
        jax.Array,
        SampleEndogenousIndividualState,
        SampleEndogenousAggregateState,
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
        next_z = self.params.rho * aggregate_s.z + self.params.nu * jax.random.normal(
            key
        )
        next_interest_rate, next_wage = self._prices(next_individual_s, next_z)
        next_time = aggregate_s.time + 1
        next_aggregate_s = SampleEndogenousAggregateState(
            z=next_z, time=next_time, interest_rate=next_interest_rate, wage=next_wage
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
        jax.Array, SampleEndogenousIndividualState, SampleEndogenousAggregateState
    ]:

        # --- reset rng ---
        reset_rng = jax.random.split(key, self.params.n_agents)

        # --- common noise ---
        z = 0.0

        # --- dummy aggregate state ---
        dummy_interest_rate = 0.0
        dummy_wage = 0.0
        dummy_aggregate_s = SampleEndogenousAggregateState(
            z=z, time=0, interest_rate=dummy_interest_rate, wage=dummy_wage
        )

        # --- sample individual states using dummy aggregate state ---
        individual_s = jax.vmap(self.sa_reset_env, in_axes=(0, None))(
            reset_rng, dummy_aggregate_s
        )

        # --- update aggregate state using individual states ---
        interest_rate, wage = self._prices(individual_s, z)
        aggregate_s = SampleEndogenousAggregateState(
            z=z, time=0, interest_rate=interest_rate, wage=wage
        )

        # --- get observations of updated aggregate state for each individual agent ---
        individual_obs = jax.vmap(self.get_individual_obs, in_axes=(0, None))(
            individual_s, aggregate_s
        )

        return individual_obs, individual_s, aggregate_s

    def _single_idio_step(
        self,
        key: jax.Array,
        individual_s: SampleEndogenousIndividualState,
        action: jax.Array,
        aggregate_s: SampleEndogenousAggregateState,
    ) -> tuple[SampleEndogenousIndividualState]:

        assert individual_s.state.ndim == 1, (
            "individual_s must be a 1D array of shape (2,)"
        )
        assert action.ndim in (0, 1), f"action ndim ({action.ndim}) must be 0 or 1"

        # --- convert to (clipped) continuous action ---
        if action.ndim == 0:
            action = self.params.discrete_actions[action]
        else:
            action = jnp.clip(action.squeeze(), 0.0, 1.0)

        # --- step single agent forward ---
        deterministic_next_state = self._single_step(
            individual_s.state, action, aggregate_s
        )
        next_wealth = deterministic_next_state[0]

        # --- idiosyncratic noise ---
        income = individual_s.state[1]
        income_idx = jnp.argmin(
            jnp.abs(self.params.discrete_states[1] - income)
        ).astype(jnp.int32)
        delta = jax.random.choice(
            key, self.params.idio_atoms, p=self.params.idio_atoms_probs
        )
        delta = delta * jnp.asarray(self.params.idio_noise, dtype=delta.dtype)
        idio_next_income_idx = jnp.clip(
            income_idx + delta, 0, self.params.num_states[1] - 1
        ).astype(jnp.int32)
        idio_next_income = self.params.discrete_states[1][idio_next_income_idx]

        # --- return next individual state ---
        next_individual_s = SampleEndogenousIndividualState(
            state=jnp.array([next_wealth, idio_next_income])
        )
        return next_individual_s

    def _single_idio_reward(
        self,
        individual_s: SampleEndogenousIndividualState,
        action: jax.Array,
        aggregate_s: SampleEndogenousAggregateState,
        next_aggregate_s: SampleEndogenousAggregateState,
    ) -> tuple[jax.Array, jax.Array]:
        assert individual_s.state.ndim == 1, (
            "individual_s must be a 1D array of shape (2,)"
        )
        assert action.ndim in (0, 1), f"action ndim ({action.ndim}) must be 0 or 1"

        # --- convert to (clipped) continuous action ---
        if action.ndim == 0:
            action = self.params.discrete_actions[action]
        else:
            action = jnp.clip(action.squeeze(), 0.0, 1.0)

        # --- calculate reward ---
        return self._single_reward(
            individual_s.state, action, aggregate_s, next_aggregate_s
        )

    def sa_step_env(
        self,
        key: jax.Array,
        individual_s: SampleEndogenousIndividualState,
        action: jax.Array,
        aggregate_s: SampleEndogenousAggregateState,
        next_aggregate_s: SampleEndogenousAggregateState,
    ) -> tuple[SampleEndogenousIndividualState, jax.Array, jax.Array]:

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
        self, key: jax.Array, aggregate_s: SampleEndogenousAggregateState
    ) -> SampleEndogenousIndividualState:

        # --- initial mean-field distribution ---
        mu_0 = jnp.ones(self.params.n_states) / self.params.n_states

        state = jax.random.choice(key, self.params.states, p=mu_0)
        return SampleEndogenousIndividualState(state=state, time=0)

    def get_individual_obs(
        self,
        individual_s: SampleEndogenousIndividualState,
        aggregate_s: SampleEndogenousAggregateState,
    ) -> jax.Array:
        return jnp.array([aggregate_s.interest_rate, aggregate_s.wage])

    def normalize_obs(
        self, aggregate_obs: jax.Array, normalize_obs: bool = False
    ) -> jax.Array:
        return jax.lax.select(
            normalize_obs, aggregate_obs, aggregate_obs.astype(jnp.float32)
        )

    def normalize_individual_s(
        self, individual_states: jax.Array, normalize_states: bool = False
    ) -> jax.Array:
        if not normalize_states:
            return individual_states

        D = self.params.pivots.shape[0]
        assert individual_states.shape[-1] == D, (
            f"expected last dimension to be {D}, got {individual_states.shape[-1]}"
        )

        # --- geometric normalization ---
        ratio = (self.params.upper_bound + self.params.pivots) / (
            self.params.lower_bound + self.params.pivots
        )
        ratio_is_small = jnp.isclose(ratio, 1.0)
        x_shifted = individual_states + self.params.pivots
        frac = jnp.clip(
            x_shifted / (self.params.lower_bound + self.params.pivots), 1e-12, None
        )
        geom_denom = jnp.where(ratio_is_small, 1.0, jnp.log(ratio))
        geom_u = jnp.log(frac) / geom_denom

        # --- use linear normalization when geometric formula is ill conditioned ---
        width = self.params.upper_bound - self.params.lower_bound
        safe_width = jnp.where(width == 0, 1.0, width)
        linear_raw = (individual_states - self.params.lower_bound) / safe_width
        linear_u = jnp.where(
            jnp.isclose(self.params.upper_bound, self.params.lower_bound),
            0.0,
            linear_raw,
        )

        # --- pick per-dimension formula and clip to [0, 1] ---
        u = jnp.where(ratio_is_small, linear_u, geom_u)
        return jnp.clip(u, 0.0, 1.0)
