import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from mfax.envs.base.base import BaseEnvironment, BaseEnvParams, BaseAggregateState


@struct.dataclass
class BaseBeachBar1DAggregateState(BaseAggregateState):
    mu: jax.Array
    bar_loc: jax.Array


@struct.dataclass
class BaseBeachBar1DEnvParams(BaseEnvParams):
    # states per dimension
    num_states: int = 101

    # actions
    actions: jax.Array = jnp.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

    # idiosyncratic noise parameters
    idio_noise: bool = True
    idio_atoms: jax.Array = jnp.array([-2, -1, 0, 1, 2])
    idio_atoms_probs: jax.Array = jnp.array([0.05, 0.1, 0.7, 0.1, 0.05])

    # common noise parameters
    common_noise: bool = False

    # terminal / truncation parameters
    max_steps_in_episode: int = 30

    def __post_init__(self):
        discrete_states = jnp.arange(self.num_states, dtype=jnp.int32)
        states = discrete_states.reshape(-1, 1)

        object.__setattr__(self, "discrete_states", discrete_states)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "n_actions", int(len(self.actions)))


class BaseBeachBar1DEnvironment(BaseEnvironment):
    def __init__(self, params: BaseBeachBar1DEnvParams = BaseBeachBar1DEnvParams()):
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

    def _project_to_legal(
        self, prev_state: jax.Array, proposed_state: jax.Array, bar_loc: jax.Array
    ) -> jax.Array:
        from_left = (prev_state < bar_loc) & (proposed_state >= bar_loc)
        from_right = (prev_state > bar_loc) & (proposed_state <= bar_loc)
        return jax.lax.select(
            from_right,
            bar_loc + 1,
            jax.lax.select(from_left, bar_loc - 1, proposed_state),
        )

    def _single_step(
        self,
        state: jax.Array,
        action: jax.Array,
        aggregate_s: BaseBeachBar1DAggregateState,
    ) -> jax.Array:
        """
        Returns next state indices and probabilities for the linear-quadratic dynamics.
        """
        mu_next_state_idx = self._project_to_legal(
            state,
            jnp.clip(state + action, 0, self.params.num_states - 1),
            aggregate_s.bar_loc,
        ).astype(jnp.int32)

        return mu_next_state_idx

    def _single_reward(
        self,
        state: jax.Array,
        action: jax.Array,
        aggregate_s: BaseBeachBar1DAggregateState,
        next_aggregate_s: BaseBeachBar1DAggregateState,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Reward function for the Beach Bar 1D environment.
        Reward penalizes being far from the bar if the bar is open, and being next to the bar when it closes. Partially observability checks whether agents can apprehend that the bar will close.
        Penalty for being in crowded areas is higher when the bar is closed.
        """

        s = jnp.asarray(state).reshape(-1)[0].astype(jnp.float32)
        a = jnp.asarray(action).reshape(-1)[0].astype(jnp.float32)
        r_step = (
            -jnp.abs((aggregate_s.bar_loc - s) * aggregate_s.z)
            - jnp.abs(jnp.abs(aggregate_s.bar_loc - s) == 1)
            * (next_aggregate_s.z == 0)
            * self.n_states  # strong negative reward for being next to bar when it is about to close / closed. We used next_aggregate_s to see whether agents can learn to apprehend that the bar might close, and learn to move away prior to closure.
            - (jnp.abs(a) / self.n_states)
        )
        r_term = r_step
        return r_step, r_term

    def is_truncated(self, time: int) -> jax.Array:
        return jnp.array(0)

    def is_terminal(self, time: int) -> jax.Array:
        return jnp.array(time >= self.params.max_steps_in_episode)
