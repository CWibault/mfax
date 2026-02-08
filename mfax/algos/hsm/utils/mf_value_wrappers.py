"""
Wrappers for mean-field policies.
"""

import jax.numpy as jnp
from typing import Optional, Dict, Any
import jax

from mfax.utils.nets.value import ValueNetwork
from mfax.utils.nets.base import RecurrentEncoder


# --- Recurrent Mean Field Value ---
class RecurrentMeanFieldValue:
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
        self.aggregate_encoder = RecurrentEncoder(**default_encoder_kwargs)
        self.value = ValueNetwork(**default_value_kwargs)

    @staticmethod
    def init_hidden(batch_size: int, hidden_size: int) -> jnp.ndarray:
        return RecurrentEncoder.init_hidden(batch_size, hidden_size)

    def _broadcast_aggregate_obs(
        self, individual_states: jnp.ndarray, aggregate_embedding: jnp.ndarray
    ) -> jnp.ndarray:
        # --- expand aggregate_embedding from [d,] to [1, d], and then broadcast to [N, d] ---
        if self.state_type == "states":
            assert individual_states.ndim == 2 and aggregate_embedding.ndim == 1
            return jnp.broadcast_to(
                aggregate_embedding[None, :],
                (individual_states.shape[0], aggregate_embedding.size),
            )
        else:
            assert individual_states.ndim == 1 and aggregate_embedding.ndim == 1
            return jnp.broadcast_to(
                aggregate_embedding[None, :],
                (individual_states.size, aggregate_embedding.size),
            )

    def _with_aggregate_embedding(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
        fn,
        *fn_inputs,
    ):
        in_axes = (0, 0, 0) + tuple(0 for _ in fn_inputs)

        def _step(_obs, _hidden, _done, *extra):
            new_hidden, aggregate_embedding = self.aggregate_encoder.apply(
                {"params": params["encoder"]},
                _hidden,
                _obs,
                _done,
            )
            broadcasted_aggregate = self._broadcast_aggregate_obs(
                individual_states, aggregate_embedding
            )
            out = fn(broadcasted_aggregate, *extra)
            return out, new_hidden

        return jax.vmap(_step, in_axes=in_axes)(
            aggregate_obs, aggregate_hidden_state, aggregate_done, *fn_inputs
        )

    def __call__(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
    ):
        value_params = {"params": params["value"]}

        def fn(broadcasted_aggregate):
            return self.value.apply(
                value_params, individual_states, broadcasted_aggregate
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
        )

    def init(
        self,
        rng: jnp.ndarray,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
    ):
        rng_enc, rng_val = jax.random.split(rng)
        encoder_params = self.aggregate_encoder.init(
            rng_enc, aggregate_hidden_state, aggregate_obs, aggregate_done
        )["params"]

        # Compute one embedding to size the value inputs.
        _, init_embeddings = self.aggregate_encoder.apply(
            {"params": encoder_params},
            aggregate_hidden_state,
            aggregate_obs,
            aggregate_done,
        )
        broadcasted = self._broadcast_aggregate_obs(
            individual_states, init_embeddings[0]
        )
        value_params = self.value.init(rng_val, individual_states, broadcasted)[
            "params"
        ]
        return {"encoder": encoder_params, "value": value_params}


class MeanFieldValue:
    def __init__(
        self,
        state_type: str,
        num_states: int | None = None,
        value_kwargs: Optional[Dict[str, Any]] = None,
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

        self.value = ValueNetwork(**default_value_kwargs)

    def _broadcast_aggregate_obs(
        self, individual_states: jnp.ndarray, aggregate_embedding: jnp.ndarray
    ) -> jnp.ndarray:
        # --- expand aggregate_embedding from [d,] to [1, d], and then broadcast to [N, d] ---
        if self.state_type == "states":
            assert individual_states.ndim == 2 and aggregate_embedding.ndim == 1
            return jnp.broadcast_to(
                aggregate_embedding[None, :],
                (individual_states.shape[0], aggregate_embedding.size),
            )
        else:
            assert individual_states.ndim == 1 and aggregate_embedding.ndim == 1
            return jnp.broadcast_to(
                aggregate_embedding[None, :],
                (individual_states.size, aggregate_embedding.size),
            )

    def _with_broadcasted_aggregate(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        fn,
        *fn_inputs,
    ):
        in_axes = (0,) + tuple(0 for _ in fn_inputs)

        def _step(_obs, *extra):
            broadcasted_aggregate = self._broadcast_aggregate_obs(
                individual_states, _obs
            )
            out = fn(broadcasted_aggregate, *extra)
            return out

        return jax.vmap(_step, in_axes=in_axes)(aggregate_obs, *fn_inputs)

    def __call__(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
    ):
        def fn(broadcasted_aggregate):
            return self.value.apply(params, individual_states, broadcasted_aggregate)

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn
        )

    def init(
        self,
        rng: jnp.ndarray,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
    ):
        broadcasted = self._broadcast_aggregate_obs(individual_states, aggregate_obs[0])
        return self.value.init(rng, individual_states, broadcasted)
