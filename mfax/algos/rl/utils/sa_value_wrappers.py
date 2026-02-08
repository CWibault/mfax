"""
Wrappers for single-agent value networks.
Wrappers are designed so that the returned parameter dict has the same structure as
the RecurrentMeanFieldValue wrapper in "mfax/algos/hsm/utils/mf_value_wrappers.py".
"""

from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp

from mfax.utils.nets.base import RecurrentEncoder
from mfax.utils.nets.value import ValueNetwork


class RecurrentSingleAgentValue:
    """
    Single-agent value wrapper that prepends a "RecurrentEncoder" to a value network.
    individual_obs_t -> RecurrentEncoder(hidden_{t-1}, individual_obs_t, done_t) -> embedding_t
    (individual_state_t, embedding_t) -> ValueNetwork -> V_t
    The returned params dict is {"encoder": ..., "value": ...} so that it is
    compatible with the "RecurrentMeanFieldValue" wrapper.
    """

    def __init__(
        self,
        state_type: str,
        num_states: int | None = None,
        value_kwargs: Optional[Dict[str, Any]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.state_type = state_type  # "states" or "indices"
        self.num_states = num_states

        default_value_kwargs = dict(
            activation="tanh",
            hidden_layer_sizes=(64, 64, 64),
            state_type=self.state_type,
            num_states=self.num_states,
        )
        if value_kwargs:
            default_value_kwargs.update(value_kwargs)

        default_encoder_kwargs = dict(hidden_size=64, embed_size=64, activation="tanh")
        if encoder_kwargs:
            default_encoder_kwargs.update(encoder_kwargs)
        self.encoder = RecurrentEncoder(**default_encoder_kwargs)
        self.value = ValueNetwork(**default_value_kwargs)

    @staticmethod
    def init_hidden(batch_size: int, hidden_size: int) -> jnp.ndarray:
        return RecurrentEncoder.init_hidden(batch_size, hidden_size)

    def _encode_obs(
        self,
        params: dict,
        individual_obs: jnp.ndarray,
        individual_hidden_state: jnp.ndarray,
        done: jnp.ndarray,
    ):
        return self.encoder.apply(
            {"params": params["encoder"]},
            individual_hidden_state,
            individual_obs,
            done,
        )

    def __call__(
        self,
        params: dict,
        individual_s: jnp.ndarray,
        individual_obs: jnp.ndarray,
        individual_hidden_state: jnp.ndarray,
        done: jnp.ndarray,
    ):
        value_params = {"params": params["value"]}
        new_hidden, obs_embedding = self._encode_obs(
            params, individual_obs, individual_hidden_state, done
        )
        out = self.value.apply(value_params, individual_s, obs_embedding)
        return out, new_hidden

    def init(
        self,
        rng: jnp.ndarray,
        individual_s: jnp.ndarray,
        individual_obs: jnp.ndarray,
        individual_hidden_state: jnp.ndarray | None,
        done: jnp.ndarray,
    ):
        if individual_hidden_state is None:
            individual_hidden_state = self.init_hidden(
                individual_s.shape[0], self.encoder.hidden_size
            )
        rng_enc, rng_val = jax.random.split(rng)
        encoder_params = self.encoder.init(
            rng_enc, individual_hidden_state, individual_obs, done
        )["params"]

        _, init_embeddings = self.encoder.apply(
            {"params": encoder_params},
            individual_hidden_state,
            individual_obs,
            done,
        )

        value_params = self.value.init(rng_val, individual_s, init_embeddings)["params"]

        return {"encoder": encoder_params, "value": value_params}
