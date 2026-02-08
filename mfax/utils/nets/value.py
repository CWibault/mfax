from typing import Sequence
import jax.numpy as jnp

from flax import linen as nn

from mfax.utils.nets.base import MLP


class ValueNetwork(nn.Module):
    hidden_layer_sizes: Sequence[int]
    activation: str
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
        self.value_head = nn.Dense(1)

    def __call__(self, state, obs):
        if self.state_type == "states":
            assert obs.ndim == state.ndim, (
                f"obs.ndim ({obs.ndim}) must match state.ndim ({state.ndim})"
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
        return self.value_head(features).squeeze(1)
