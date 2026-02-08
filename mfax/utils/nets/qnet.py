from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from mfax.utils.nets.base import MLP


class DiscreteQNet(nn.Module):
    """
    Q-network for discrete actions.
    """

    n_actions: int
    hidden_layer_sizes: tuple[int, ...]
    activation: Callable
    tau: float
    alpha: float
    state_type: str
    num_states: int | None = None

    def setup(self):
        if self.state_type == "states":
            self.state_embedding = nn.Dense(self.hidden_layer_sizes[0] // 2)
        elif self.state_type == "indices":
            self.state_embedding = nn.Embed(
                self.num_states, self.hidden_layer_sizes[0] // 2
            )
        else:
            raise ValueError(f"Invalid state type: {self.state_type}")
        self.obs_embedding = nn.Dense(self.hidden_layer_sizes[0] // 2)
        self.features = MLP(self.hidden_layer_sizes[1:], self.activation)
        self.dense = nn.Dense(self.n_actions)

    def __call__(self, state, obs):
        if self.state_type == "states":
            assert obs.ndim == state.ndim, (
                f"obs.ndim ({obs.ndim}) must be one more than state.ndim ({state.ndim})"
            )
        else:
            assert obs.ndim == state.ndim + 1, (
                f"obs.ndim ({obs.ndim}) must be one more than state.ndim ({state.ndim})"
            )
        state_embedding = self.state_embedding(state)
        obs_embedding = self.obs_embedding(obs)
        features = self.features(
            jnp.concatenate([state_embedding, obs_embedding], axis=-1)
        )
        return self.dense(features)

    def softmax(self, state, obs):
        q_vals = self(state, obs)
        logits = q_vals / self.tau
        prob_a = jax.nn.softmax(logits, axis=-1)
        return q_vals, prob_a

    def epsilon_greedy(self, state, obs, eps, rng):
        rng_a, rng_e = jax.random.split(rng, 2)
        q_vals = self(state, obs)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        random_actions = jax.random.randint(
            rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
        )
        action = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            random_actions,
            greedy_actions,
        )
        return action

    def argmax(self, state, obs):
        q_vals = self(state, obs)
        action = jnp.argmax(q_vals, axis=-1)
        return q_vals, action

    def sample_softmax(self, state, obs, rng):
        q_vals, prob_a = self.softmax(state, obs)
        logits = jnp.log(prob_a + 1e-12)
        action = jax.random.categorical(rng, logits)
        return q_vals, action

    def take_action(self, state, obs, action_idx):
        q_values = self(state, obs)
        return jnp.take_along_axis(q_values, action_idx[..., None], axis=-1).squeeze(-1)
