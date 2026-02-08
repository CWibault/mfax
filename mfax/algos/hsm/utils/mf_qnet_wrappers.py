"""
Wrappers for mean-field Q-networks.
"""

from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp

from mfax.utils.nets.qnet import DiscreteQNet
from mfax.utils.nets.base import RecurrentEncoder


class RecurrentMeanFieldQNet:
    def __init__(
        self,
        state_type: str,
        num_states: int | None = None,
        q_net_type: str = "discrete",
        q_net_kwargs: Optional[Dict[str, Any]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.state_type = state_type  # "states" or "indices"
        self.num_states = num_states

        default_q_net_kwargs = dict(
            activation="tanh",
            hidden_layer_sizes=(64, 64, 64),
            n_actions=1,
            state_type=self.state_type,
            num_states=self.num_states,
        )
        if q_net_kwargs:
            default_q_net_kwargs.update(q_net_kwargs)

        default_encoder_kwargs = dict(hidden_size=64, embed_size=64, activation="tanh")
        if encoder_kwargs:
            default_encoder_kwargs.update(encoder_kwargs)
        self.aggregate_encoder = RecurrentEncoder(**default_encoder_kwargs)
        if q_net_type == "discrete":
            self.q_net = DiscreteQNet(**default_q_net_kwargs)
        else:
            raise ValueError(f"Invalid q_net_type: {q_net_type}. Expected 'discrete'.")

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
        q_params = {"params": params["q_net"]}

        def fn(broadcasted_aggregate):
            return self.q_net.apply(q_params, individual_states, broadcasted_aggregate)

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
        rng_enc, rng_q = jax.random.split(rng)
        encoder_params = self.aggregate_encoder.init(
            rng_enc, aggregate_hidden_state, aggregate_obs, aggregate_done
        )["params"]

        # Compute one embedding to size the Q-net inputs.
        _, init_embeddings = self.aggregate_encoder.apply(
            {"params": encoder_params},
            aggregate_hidden_state,
            aggregate_obs,
            aggregate_done,
        )
        broadcasted = self._broadcast_aggregate_obs(
            individual_states, init_embeddings[0]
        )
        q_params = self.q_net.init(rng_q, individual_states, broadcasted)["params"]
        return {"encoder": encoder_params, "q_net": q_params}

    def softmax(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
    ):
        q_params = {"params": params["q_net"]}

        def fn(broadcasted_aggregate):
            return self.q_net.apply(
                q_params, individual_states, broadcasted_aggregate, method="softmax"
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
        )

    def epsilon_greedy(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
        eps: float,
        rng: jnp.ndarray,
    ):
        q_params = {"params": params["q_net"]}
        rngs = jax.random.split(rng, aggregate_obs.shape[0])

        def fn(broadcasted_aggregate, rng_i):
            return self.q_net.apply(
                q_params,
                individual_states,
                broadcasted_aggregate,
                eps,
                rng_i,
                method="epsilon_greedy",
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
            rngs,
        )

    def argmax(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
    ):
        q_params = {"params": params["q_net"]}

        def fn(broadcasted_aggregate):
            return self.q_net.apply(
                q_params, individual_states, broadcasted_aggregate, method="argmax"
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
        )

    def sample_softmax(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
        rng: jnp.ndarray,
    ):
        q_params = {"params": params["q_net"]}
        rngs = jax.random.split(rng, aggregate_obs.shape[0])

        def fn(broadcasted_aggregate, rng_i):
            return self.q_net.apply(
                q_params,
                individual_states,
                broadcasted_aggregate,
                rng_i,
                method="sample_softmax",
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
            rngs,
        )

    def take_action(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
        action_idxs: jnp.ndarray,
    ):
        q_params = {"params": params["q_net"]}

        def fn(broadcasted_aggregate, _action_idxs):
            return self.q_net.apply(
                q_params,
                individual_states,
                broadcasted_aggregate,
                _action_idxs,
                method="take_action",
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
            action_idxs,
        )


class MeanFieldQNet:
    def __init__(
        self,
        state_type: str,
        num_states: int | None = None,
        q_net_type: str = "discrete",
        q_net_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.state_type = state_type  # "states" or "indices"
        self.num_states = num_states

        default_q_net_kwargs = dict(
            activation="tanh",
            hidden_layer_sizes=(64, 64, 64),
            n_actions=1,
            state_type=self.state_type,
            num_states=self.num_states,
        )
        if q_net_kwargs:
            default_q_net_kwargs.update(q_net_kwargs)
        if q_net_type == "discrete":
            self.q_net = DiscreteQNet(**default_q_net_kwargs)
        else:
            raise ValueError(f"Invalid q_net_type: {q_net_type}. Expected 'discrete'.")

    def _broadcast_aggregate_obs(
        self, individual_states: jnp.ndarray, aggregate_embedding: jnp.ndarray
    ) -> jnp.ndarray:
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
            return self.q_net.apply(params, individual_states, broadcasted_aggregate)

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
        return self.q_net.init(rng, individual_states, broadcasted)

    def softmax(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
    ):
        def fn(broadcasted_aggregate):
            return self.q_net.apply(
                params, individual_states, broadcasted_aggregate, method="softmax"
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn
        )

    def epsilon_greedy(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        eps: float,
        rng: jnp.ndarray,
    ):
        rngs = jax.random.split(rng, aggregate_obs.shape[0])

        def fn(broadcasted_aggregate, rng_i):
            return self.q_net.apply(
                params,
                individual_states,
                broadcasted_aggregate,
                eps,
                rng_i,
                method="epsilon_greedy",
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn, rngs
        )

    def argmax(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
    ):
        def fn(broadcasted_aggregate):
            return self.q_net.apply(
                params, individual_states, broadcasted_aggregate, method="argmax"
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn
        )

    def sample_softmax(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        rng: jnp.ndarray,
    ):
        rngs = jax.random.split(rng, aggregate_obs.shape[0])

        def fn(broadcasted_aggregate, rng_i):
            return self.q_net.apply(
                params,
                individual_states,
                broadcasted_aggregate,
                rng_i,
                method="sample_softmax",
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn, rngs
        )

    def take_action(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        action_idxs: jnp.ndarray,
    ):
        def fn(broadcasted_aggregate, _action_idxs):
            return self.q_net.apply(
                params,
                individual_states,
                broadcasted_aggregate,
                _action_idxs,
                method="take_action",
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn, action_idxs
        )
