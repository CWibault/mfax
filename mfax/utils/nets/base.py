from typing import Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp

ACTIVATIONS = {
    "relu": jax.nn.relu,
    "swish": jax.nn.swish,
    "tanh": jax.nn.tanh,
    "gelu": jax.nn.gelu,
}


class MLP(nn.Module):
    hidden_layer_sizes: Sequence[int]
    activation: str

    @nn.compact
    def __call__(self, x):
        activation = ACTIVATIONS[self.activation]
        x = x.reshape((x.shape[0], -1))
        for size in self.hidden_layer_sizes:
            x = nn.Dense(size)(x)
            x = activation(x)
        return x


class RecurrentEncoder(nn.Module):
    hidden_size: int
    embed_size: int
    activation: str

    @nn.compact
    def __call__(self, hidden: jnp.ndarray, obs: jnp.ndarray, done: jnp.ndarray):
        # --- reset hidden state for terminated environments ---
        hidden = jnp.where(done[..., None], jnp.zeros_like(hidden), hidden)
        new_hidden, _ = nn.GRUCell(self.hidden_size)(hidden, obs)
        embedding = nn.Dense(self.embed_size)(ACTIVATIONS[self.activation](new_hidden))
        return new_hidden, embedding

    @staticmethod
    def init_hidden(batch_size: int, hidden_size: int) -> jnp.ndarray:
        return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
