import jax
import jax.numpy as jnp
from flax import struct
import math
from gymnax.environments import spaces

from mfax.envs.base.base import BaseEnvironment, BaseEnvParams, BaseAggregateState


@struct.dataclass
class BaseLinearQuadraticAggregateState(BaseAggregateState):
    mu_mean: jax.Array


@struct.dataclass
class BaseLinearQuadraticEnvParams(BaseEnvParams):
    # parameters as in Wu et al. (2025) and Perrin et al. (2020)
    kappa: float = 0.5
    c_action: float = 0.5
    q: float = 0.1
    c_term: float = 1.0
    sigma: float = 1.0
    rho: float = 0.5

    # states per dimension
    num_states: int = 99

    # actions
    actions: jax.Array = jnp.array([-3, -2, -1, 0, 1, 2, 3])

    # idiosyncratic noise parameters
    idio_noise: bool = True
    idio_atoms_per_side: int = 3
    idio_atoms: jnp.ndarray = jnp.empty((0,))
    idio_atoms_probs: jnp.ndarray = jnp.empty((0,))

    # common noise parameters
    common_noise: bool = False

    # terminal / truncation parameters
    max_steps_in_episode: int = 30

    def __post_init__(self):
        discrete_states = jnp.arange(self.num_states, dtype=jnp.int32)
        states = discrete_states.reshape(-1, 1)

        def gaussian_pdf(z):
            return jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * math.pi)

        idio_atoms = jnp.arange(
            -self.idio_atoms_per_side, self.idio_atoms_per_side + 1, dtype=jnp.float32
        )
        idio_atoms_probs = gaussian_pdf(idio_atoms)
        idio_atoms_probs = idio_atoms_probs / idio_atoms_probs.sum()

        object.__setattr__(self, "discrete_states", discrete_states)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "n_actions", int(len(self.actions)))
        object.__setattr__(self, "idio_atoms", idio_atoms)
        object.__setattr__(self, "idio_atoms_probs", idio_atoms_probs)


class BaseLinearQuadraticEnvironment(BaseEnvironment):
    def __init__(
        self, params: BaseLinearQuadraticEnvParams = BaseLinearQuadraticEnvParams()
    ):
        super().__init__(params)

    @property
    def is_partially_observable(self) -> bool:
        return True

    @property
    def n_actions(self) -> int:
        return int(self.params.n_actions)

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.n_actions)

    @property
    def individual_s_dim(self) -> int:
        return 1

    def _single_step(
        self,
        state: jax.Array,
        action: jax.Array,
        aggregate_s: BaseLinearQuadraticAggregateState,
    ) -> jax.Array:
        """
        Returns next state indices and probabilities for the linear-quadratic dynamics.
        """
        # --- common noise: deterministic piecewise effect ---
        eps = (
            jax.lax.select(
                aggregate_s.time < 8,
                aggregate_s.z * -10,
                jax.lax.select(
                    aggregate_s.time > 20,
                    aggregate_s.z * 10,
                    jnp.array(0, dtype=aggregate_s.z.dtype),
                ),
            )
            * self.params.common_noise
        )

        mu_next_state_idx = jnp.clip(
            state + action + self.params.sigma * (self.params.rho * eps),
            0,
            self.params.num_states - 1,
        ).astype(jnp.int32)
        return mu_next_state_idx

    def _single_reward(
        self,
        state: jax.Array,
        action: jax.Array,
        aggregate_s: BaseLinearQuadraticAggregateState,
        next_aggregate_s: BaseLinearQuadraticAggregateState,
    ) -> tuple[jax.Array, jax.Array]:

        s = jnp.asarray(state).reshape(-1)[0].astype(jnp.float32)
        a = jnp.asarray(action).reshape(-1)[0].astype(jnp.float32)
        r_step = (
            -self.params.c_action * (a**2)
            + self.params.q * a * (next_aggregate_s.mu_mean - s)
            - (self.params.kappa / 2) * ((next_aggregate_s.mu_mean - s) ** 2)
        )
        r_term = -1 * (self.params.c_term / 2) * ((next_aggregate_s.mu_mean - s) ** 2)
        return r_step, r_term

    def is_truncated(self, time: int) -> jax.Array:
        return jnp.array(0)

    def is_terminal(self, time: int) -> jax.Array:
        return jnp.array(time >= self.params.max_steps_in_episode)
