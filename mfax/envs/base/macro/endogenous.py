import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from mfax.envs.base.base import BaseEnvironment, BaseAggregateState, BaseEnvParams


@struct.dataclass
class BaseEndogenousAggregateState(BaseAggregateState):
    interest_rate: jax.Array
    wage: jax.Array


@struct.dataclass
class BaseEndogenousEnvParams(BaseEnvParams):
    # reward parameters
    sigma: float = 2.0

    # shock parameters
    rho: float = 0.9
    nu: float = 0.03

    # prices
    cobb_douglas_alpha: float = 0.36

    # idiosyncratic noise parameters
    idio_noise: bool = True

    # common noise parameters
    common_noise: bool = True

    # terminal / truncation parameters
    max_steps_in_episode: int = 128

    # action space
    discrete_n_actions: int = 20

    def __post_init__(self):
        actions = 0.5 * (
            jnp.linspace(0, 1, self.discrete_n_actions + 1)[:-1]
            + jnp.linspace(0, 1, self.discrete_n_actions + 1)[1:]
        )
        object.__setattr__(self, "discrete_actions", actions)


class BaseEndogenousEnvironment(BaseEnvironment):
    def __init__(self, params: BaseEndogenousEnvParams = BaseEndogenousEnvParams()):
        super().__init__(params)

    @property
    def is_partially_observable(self) -> bool:
        return True

    @property
    def action_space(self) -> spaces.Box:
        """Action space for the environment."""
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=jnp.float32,
        )

    @property
    def n_actions(self) -> int:
        """Number of actions possible in the environment."""
        return self.params.discrete_n_actions

    @property
    def discrete_action_space(self) -> spaces.Discrete:
        """Action space for the environment."""
        return spaces.Discrete(self.n_actions)

    @property
    def individual_s_dim(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        return 2

    def _wealth_idx_income_idx_to_state_idx(self, wealth_idx, income_idx):
        wealth_idx = jnp.clip(jnp.int32(wealth_idx), 0, self.params.num_states[0] - 1)
        income_idx = jnp.clip(jnp.int32(income_idx), 0, self.params.num_states[1] - 1)
        return wealth_idx * self.params.num_states[1] + income_idx

    def _state_idx_to_wealth_idx_income_idx(self, state_idx: int) -> tuple[int, int]:
        state_idx = jnp.clip(jnp.int32(state_idx), 0, self.params.n_states - 1)
        wealth_idx = state_idx // self.params.num_states[1]
        income_idx = state_idx % self.params.num_states[1]
        return wealth_idx, income_idx

    def _single_step(
        self, state: jax.Array, action: jax.Array, aggregate_s: BaseAggregateState
    ) -> tuple[jax.Array, jax.Array]:
        """
        Returns deterministic next state for a current state, action and aggregate state (i.e. no idiosyncratic noise)
        """
        # --- overall wealth (including income, wage, interest rate) ---
        wealth = (1.0 + aggregate_s.interest_rate) * state[
            0
        ] + aggregate_s.wage * state[1]

        # --- deterministic next state given action (environment specific) ---
        next_wealth = wealth * (1 - action)
        return jnp.array([next_wealth, state[1]])

    def _single_reward(
        self,
        state: jax.Array,
        action: jax.Array,
        aggregate_s: BaseAggregateState,
        next_aggregate_s: BaseAggregateState,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Returns reward for a current state, action and aggregate state.
        """
        # --- overall wealth (including income, wage, interest rate) ---
        wealth = (1.0 + aggregate_s.interest_rate) * state[
            0
        ] + aggregate_s.wage * state[1]

        # --- deterministic next wealth given action (environment specific) ---
        next_wealth = wealth * (1.0 - action)

        # --- calculate reward ---
        consumption = wealth - next_wealth
        r_step = (consumption ** (1.0 - self.params.sigma)) / (1.0 - self.params.sigma)
        r_term = (wealth ** (1.0 - self.params.sigma)) / (1.0 - self.params.sigma)
        return r_step, r_term

    def is_truncated(self, time: int) -> jax.Array:
        return jnp.array(0)

    def is_terminal(self, time: int) -> jax.Array:
        return jnp.array(time >= self.params.max_steps_in_episode)
