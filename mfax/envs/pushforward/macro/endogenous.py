import jax
import jax.numpy as jnp
from flax import struct
from dataclasses import field

from mfax.envs.base.macro.endogenous import (
    BaseEndogenousEnvironment,
    BaseEndogenousEnvParams,
    BaseEndogenousAggregateState,
)
from mfax.envs.pushforward.base import (
    PushforwardEnvironment,
    PushforwardEnvParams,
    PushforwardAggregateState,
)
from mfax.envs.pushforward.utils import distribute


@struct.dataclass
class PushforwardEndogenousAggregateState(
    PushforwardAggregateState, BaseEndogenousAggregateState
):
    pass


@struct.dataclass
class PushforwardEndogenousEnvParams(PushforwardEnvParams, BaseEndogenousEnvParams):
    # states
    states: jax.Array = field(default_factory=lambda: jnp.empty((0, 0)))

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


class PushforwardEndogenousEnvironment(
    PushforwardEnvironment, BaseEndogenousEnvironment
):
    def _prices(self, mu: jax.Array, z: float) -> tuple[float, float]:
        """
        Calculates the endogenous prices (i.e. interest rate and wage from aggregate state)
        """

        # --- sum over nuisance variable to obtain marginal distributions ---
        mu = jnp.reshape(mu, (self.params.num_states[0], self.params.num_states[1]))
        av_wealth = jnp.sum(mu.sum(1) * self.params.discrete_states[0])
        av_income = jnp.sum(mu.sum(0) * self.params.discrete_states[1])

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

    def mf_reset_env(
        self, key: jax.Array
    ) -> tuple[jax.Array, PushforwardEndogenousAggregateState]:

        # --- initial mean-field distribution ---
        mu_0 = jnp.ones(self.params.n_states) / self.params.n_states

        # --- common noise ---
        z = 0.0

        # --- calculate prices to avoid re-calculating ---
        interest_rate, wage = self._prices(mu_0, z)
        aggregate_s = PushforwardEndogenousAggregateState(
            mu=mu_0, z=z, time=0, interest_rate=interest_rate, wage=wage
        )

        return self.get_shared_obs(aggregate_s), aggregate_s

    def mf_step_env(
        self,
        key: jax.Array,
        aggregate_s: PushforwardEndogenousAggregateState,
        prob_a: jax.Array,
    ) -> tuple[
        jax.Array, PushforwardEndogenousAggregateState, jax.Array, jax.Array, jax.Array
    ]:

        # --- update aggregate state: mean-field, z and time, and obtain the reward associated with each state-action pair ---
        next_mu = self.mf_transition(aggregate_s.mu, prob_a, aggregate_s)
        next_z = self.params.rho * aggregate_s.z + self.params.nu * jax.random.normal(
            key
        )
        next_interest_rate, next_wage = self._prices(next_mu, next_z)
        next_time = aggregate_s.time + 1
        next_aggregate_s = PushforwardEndogenousAggregateState(
            mu=next_mu,
            z=next_z,
            time=next_time,
            interest_rate=next_interest_rate,
            wage=next_wage,
        )

        # --- check for termination and truncation ---
        terminated = self.is_terminal(next_time)
        truncated = self.is_truncated(next_time)

        # --- select between step and terminated reward ---
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
        state_idx: int,
        action: int,
        aggregate_s: PushforwardEndogenousAggregateState,
    ) -> tuple[jax.Array, jax.Array]:

        assert state_idx.ndim == 0, "state_idx must be a scalar"
        assert action.ndim in (0, 1), f"action ndim ({action.ndim}) must be 0 or 1"

        # --- identify state ---
        state = self.params.states[state_idx]

        # --- convert to (clipped) continuous action ---
        if action.ndim == 0:
            action = self.params.discrete_actions[action]
        else:
            action = jnp.clip(action.squeeze(), 0.0, 1.0)

        # --- deterministic next state given action (environment specific) ---
        deterministic_next_state = self._single_step(state, action, aggregate_s)
        next_wealth = deterministic_next_state[0]
        next_wealth_idxs, dist_probs = distribute(
            self.params.discrete_states[0], next_wealth
        )  # shape (2,)

        # --- idiosyncratic noise (environment specific) ---
        _, income_idx = self._state_idx_to_wealth_idx_income_idx(state_idx)
        idio_next_wealth_idxs = jnp.repeat(
            next_wealth_idxs, len(self.params.idio_atoms)
        )  # shape (2 * n_idio_atoms,)
        idio_next_income_idxs = jnp.clip(
            income_idx + self.params.idio_atoms, 0, self.params.num_states[1] - 1
        )  # shape (n_idio_atoms,)
        idio_next_income_idxs = jnp.tile(
            idio_next_income_idxs, 2
        )  # shape (2 * n_idio_atoms,)
        idio_next_state_idxs = self._wealth_idx_income_idx_to_state_idx(
            idio_next_wealth_idxs, idio_next_income_idxs
        )  # shape (2, n_idio_atoms)
        idio_atoms_probs = jnp.outer(
            dist_probs, self.params.idio_atoms_probs * self.params.idio_noise
        ).ravel()

        # --- index for mean (for the (1 - idio_noise) mass) ---
        mu_next_income_idxs = jnp.repeat(income_idx, 2)  # shape (2 * n_idio_atoms,)
        mu_next_state_idxs = self._wealth_idx_income_idx_to_state_idx(
            next_wealth_idxs, mu_next_income_idxs
        )  # shape (2, n_idio_atoms)
        mu_probs = (1.0 - self.params.idio_noise) * dist_probs

        # --- return next state indices and probabilities ---
        next_state_idxs = jnp.concatenate(
            [idio_next_state_idxs, mu_next_state_idxs], axis=0
        ).reshape(-1)  # shape (2 * n_idio_atoms + 1, 1)
        probs = jnp.concatenate([idio_atoms_probs, mu_probs], axis=0).reshape(
            -1
        )  # shape (2 * n_idio_atoms + 1, 1)
        probs = probs / jnp.where(probs.sum() > 0, probs.sum(), 1.0)
        return next_state_idxs, probs

    def _single_pushforward_reward(
        self,
        state_idx: jax.Array,
        action: int,
        aggregate_s: PushforwardEndogenousAggregateState,
        next_aggregate_s: PushforwardEndogenousAggregateState,
    ) -> tuple[jax.Array, jax.Array]:

        assert state_idx.ndim == 0, "state_idx must be a scalar"
        assert action.ndim in (0, 1), f"action ndim ({action.ndim}) must be 0 or 1"

        # --- identify state ---
        state = self.params.states[state_idx]

        # --- convert to (clipped) continuous action ---
        if action.ndim == 0:
            action = self.params.discrete_actions[action]
        else:
            action = jnp.clip(action.squeeze(), 0.0, 1.0)

        # --- calculate (expected, if depends on next state) reward ---
        r_step, r_term = self._single_reward(
            state, action, aggregate_s, next_aggregate_s
        )
        return r_step, r_term

    def get_shared_obs(
        self, aggregate_s: PushforwardEndogenousAggregateState
    ) -> jax.Array:
        return jnp.array([aggregate_s.interest_rate, aggregate_s.wage])

    def normalize_obs(
        self, shared_obs: jax.Array, normalize_obs: bool = False
    ) -> jax.Array:
        return jax.lax.select(normalize_obs, shared_obs, shared_obs.astype(jnp.float32))

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
