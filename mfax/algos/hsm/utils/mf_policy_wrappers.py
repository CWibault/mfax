"""
Wrappers for mean-field policies.
"""

import jax.numpy as jnp
from typing import Optional, Dict, Any
import jax

from mfax.utils.nets.policy import DiscretePolicy, BetaPolicy
from mfax.utils.nets.base import RecurrentEncoder


# --- Mean Field Policy ---
class RecurrentMeanFieldPolicy:
    def __init__(
        self,
        state_type: str,
        num_states: int | None = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.state_type = state_type  # "states" or "indices"
        self.num_states = num_states
        default_policy_kwargs = dict(
            activation="tanh",
            hidden_layer_sizes=(64, 64, 64),
            n_actions=1,
            state_type=self.state_type,
            num_states=self.num_states,
        )
        if policy_kwargs:
            default_policy_kwargs.update(policy_kwargs)

        default_encoder_kwargs = dict(hidden_size=64, embed_size=64, activation="tanh")
        if encoder_kwargs:
            default_encoder_kwargs.update(encoder_kwargs)
        self.aggregate_encoder = RecurrentEncoder(**default_encoder_kwargs)
        self.policy = DiscretePolicy(**default_policy_kwargs)

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
        rng: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}
        rngs = jax.random.split(rng, aggregate_obs.shape[0])

        def fn(broadcasted_aggregate, rng_i):
            return self.policy.apply(
                policy_params, individual_states, broadcasted_aggregate, rng_i
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

    def init(
        self,
        rng: jnp.ndarray,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
        rng_action: jnp.ndarray,
    ):
        rng_enc, rng_pol = jax.random.split(rng)
        encoder_params = self.aggregate_encoder.init(
            rng_enc, aggregate_hidden_state, aggregate_obs, aggregate_done
        )["params"]

        # One embedding to size policy inputs.
        _, init_embeddings = self.aggregate_encoder.apply(
            {"params": encoder_params},
            aggregate_hidden_state,
            aggregate_obs,
            aggregate_done,
        )
        broadcasted = self._broadcast_aggregate_obs(
            individual_states, init_embeddings[0]
        )
        policy_params = self.policy.init(
            rng_pol, individual_states, broadcasted, rng_action
        )["params"]
        return {"encoder": encoder_params, "policy": policy_params}

    def _action_dist(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}

        def fn(broadcasted_aggregate):
            return self.policy.apply(
                policy_params,
                individual_states,
                broadcasted_aggregate,
                method="_action_dist",
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
        )

    def sample(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
        rng: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}
        rngs = jax.random.split(rng, aggregate_obs.shape[0])

        def fn(broadcasted_aggregate, rng_i):
            return self.policy.apply(
                policy_params,
                individual_states,
                broadcasted_aggregate,
                rng_i,
                method="sample",
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

    def sample_and_log_prob(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
        rng: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}
        rngs = jax.random.split(rng, aggregate_obs.shape[0])

        def fn(broadcasted_aggregate, rng_i):
            return self.policy.apply(
                policy_params,
                individual_states,
                broadcasted_aggregate,
                rng_i,
                method="sample_and_log_prob",
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

    def dist_prob(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}

        def fn(broadcasted_aggregate):
            return self.policy.apply(
                policy_params,
                individual_states,
                broadcasted_aggregate,
                method="dist_prob",
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
        )

    def dist_prob_sample_and_log_prob(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
        rng: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}
        rngs = jax.random.split(rng, aggregate_obs.shape[0])

        def fn(broadcasted_aggregate, rng_i):
            return self.policy.apply(
                policy_params,
                individual_states,
                broadcasted_aggregate,
                rng_i,
                method="dist_prob_sample_and_log_prob",
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

    def dist_log_prob_entropy(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}

        def fn(broadcasted_aggregate):
            return self.policy.apply(
                policy_params,
                individual_states,
                broadcasted_aggregate,
                method="dist_log_prob_entropy",
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
        )

    def mode(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}

        def fn(broadcasted_aggregate):
            return self.policy.apply(
                policy_params, individual_states, broadcasted_aggregate, method="mode"
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
        )

    def log_prob(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
        actions: jnp.ndarray,
    ):
        # actions can be 2D or 3D
        policy_params = {"params": params["policy"]}

        def fn(broadcasted_aggregate, _actions):
            return self.policy.apply(
                policy_params,
                individual_states,
                broadcasted_aggregate,
                _actions,
                method="log_prob",
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
            actions,
        )

    def entropy(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}

        def fn(broadcasted_aggregate):
            return self.policy.apply(
                policy_params,
                individual_states,
                broadcasted_aggregate,
                method="entropy",
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
        )

    def log_prob_entropy(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
        actions: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}

        def fn(broadcasted_aggregate, _actions):
            return self.policy.apply(
                policy_params,
                individual_states,
                broadcasted_aggregate,
                _actions,
                method="log_prob_entropy",
            )

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
            actions,
        )


class MeanFieldPolicy:
    def __init__(
        self,
        state_type: str,
        num_states: int | None = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.state_type = state_type  # "states" or "indices"
        self.num_states = num_states
        default_policy_kwargs = dict(
            activation="tanh",
            hidden_layer_sizes=(64, 64, 64),
            n_actions=1,
            state_type=self.state_type,
            num_states=self.num_states,
        )
        if policy_kwargs:
            default_policy_kwargs.update(policy_kwargs)

        self.policy = DiscretePolicy(**default_policy_kwargs)

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
        rng: jnp.ndarray,
    ):
        rngs = jax.random.split(rng, aggregate_obs.shape[0])

        def fn(broadcasted_aggregate, rng_i):
            return self.policy.apply(
                params, individual_states, broadcasted_aggregate, rng_i
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn, rngs
        )

    def init(
        self,
        rng: jnp.ndarray,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        rng_action: jnp.ndarray,
    ):
        broadcasted = self._broadcast_aggregate_obs(individual_states, aggregate_obs[0])
        return self.policy.init(rng, individual_states, broadcasted, rng_action)

    def _action_dist(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
    ):
        return self.policy.apply(
            params, individual_states, aggregate_obs, method="_action_dist"
        )

    def sample(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        rng: jnp.ndarray,
    ):
        rngs = jax.random.split(rng, aggregate_obs.shape[0])

        def fn(broadcasted_aggregate, rng_i):
            return self.policy.apply(
                params, individual_states, broadcasted_aggregate, rng_i, method="sample"
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn, rngs
        )

    def sample_and_log_prob(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        rng: jnp.ndarray,
    ):
        rngs = jax.random.split(rng, aggregate_obs.shape[0])

        def fn(broadcasted_aggregate, rng_i):
            return self.policy.apply(
                params,
                individual_states,
                broadcasted_aggregate,
                rng_i,
                method="sample_and_log_prob",
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn, rngs
        )

    def dist_prob(
        self, params: dict, individual_states: jnp.ndarray, aggregate_obs: jnp.ndarray
    ):

        def fn(broadcasted_aggregate):
            return self.policy.apply(
                params, individual_states, broadcasted_aggregate, method="dist_prob"
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn
        )

    def dist_prob_sample_and_log_prob(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        rng: jnp.ndarray,
    ):
        rngs = jax.random.split(rng, aggregate_obs.shape[0])

        def fn(broadcasted_aggregate, rng_i):
            return self.policy.apply(
                params,
                individual_states,
                broadcasted_aggregate,
                rng_i,
                method="dist_prob_sample_and_log_prob",
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn, rngs
        )

    def dist_log_prob_entropy(
        self, params: dict, individual_states: jnp.ndarray, aggregate_obs: jnp.ndarray
    ):

        def fn(broadcasted_aggregate):
            return self.policy.apply(
                params,
                individual_states,
                broadcasted_aggregate,
                method="dist_log_prob_entropy",
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn
        )

    def mode(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
    ):

        def fn(broadcasted_aggregate):
            return self.policy.apply(
                params, individual_states, broadcasted_aggregate, method="mode"
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn
        )

    def log_prob(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        actions: jnp.ndarray,
    ):
        # actions can be 2D or 3D

        def fn(broadcasted_aggregate, _actions):
            return self.policy.apply(
                params,
                individual_states,
                broadcasted_aggregate,
                _actions,
                method="log_prob",
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn, actions
        )

    def entropy(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
    ):

        def fn(broadcasted_aggregate):
            return self.policy.apply(
                params, individual_states, broadcasted_aggregate, method="entropy"
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn
        )

    def log_prob_entropy(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        actions: jnp.ndarray,
    ):

        def fn(broadcasted_aggregate, _actions):
            return self.policy.apply(
                params,
                individual_states,
                broadcasted_aggregate,
                _actions,
                method="log_prob_entropy",
            )

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn, actions
        )


class RecurrentMeanFieldContinuousPolicy:
    def __init__(
        self,
        state_type: str,
        num_states: int | None = None,
        actions: jnp.ndarray | None = None,
        n_actions_per_dim: int | None = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # --- discretization settings ---
        # Mean Field Sequence Generation expects a matrix of probabilities for each action - i.e. prob_a over discrete actions.
        # Turn the continuous density into categorical logits by evaluating log_prob on `action_grid`, then softmaxing the logits.
        self.state_type = state_type
        self.num_states = num_states
        policy_kwargs = dict(policy_kwargs) if policy_kwargs else {}
        default_policy_kwargs = dict(
            activation="tanh",
            hidden_layer_sizes=(64, 64, 64),
            action_dim=1,
            action_range=(0.0, 1.0),
            state_type=self.state_type,
            num_states=self.num_states,
        )
        default_policy_kwargs.update(policy_kwargs)
        self.policy = BetaPolicy(**default_policy_kwargs)
        self.action_dim = int(default_policy_kwargs["action_dim"])

        if actions is None:
            assert n_actions_per_dim is not None, (
                "n_actions_per_dim must be provided if actions is not provided"
            )
            per_dim_axes = [
                jnp.linspace(
                    default_policy_kwargs["action_range"][0],
                    default_policy_kwargs["action_range"][1],
                    n_actions_per_dim,
                )
                for i in range(self.action_dim)
            ]
            mesh = jnp.meshgrid(*per_dim_axes, indexing="ij")
            self.action_grid = jnp.stack([m.reshape(-1) for m in mesh], axis=-1)
        else:
            if actions.ndim == 1:
                actions = actions.reshape(-1, self.action_dim)
            self.action_grid = actions

        default_encoder_kwargs = dict(hidden_size=64, embed_size=64, activation="tanh")
        if encoder_kwargs:
            default_encoder_kwargs.update(encoder_kwargs)
        self.aggregate_encoder = RecurrentEncoder(**default_encoder_kwargs)

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
        rng: jnp.ndarray,
    ):
        return self.dist_prob(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
        )

    def init(
        self,
        rng: jnp.ndarray,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
        rng_action: jnp.ndarray,
    ):
        rng_enc, rng_pol = jax.random.split(rng)
        encoder_params = self.aggregate_encoder.init(
            rng_enc, aggregate_hidden_state, aggregate_obs, aggregate_done
        )["params"]

        # One embedding to size policy inputs.
        _, init_embeddings = self.aggregate_encoder.apply(
            {"params": encoder_params},
            aggregate_hidden_state,
            aggregate_obs,
            aggregate_done,
        )
        broadcasted = self._broadcast_aggregate_obs(
            individual_states, init_embeddings[0]
        )
        policy_params = self.policy.init(
            rng_pol, individual_states, broadcasted, rng_action
        )["params"]
        return {"encoder": encoder_params, "policy": policy_params}

    def _action_logits(
        self,
        policy_params: dict,
        individual_states: jnp.ndarray,
        broadcasted_aggregate: jnp.ndarray,
    ) -> jnp.ndarray:
        actions = jnp.broadcast_to(
            self.action_grid[None, ...],
            (
                individual_states.shape[0],
                self.action_grid.shape[0],
                self.action_grid.shape[-1],
            ),
        )

        def _log_prob(action):
            return self.policy.apply(
                policy_params,
                individual_states,
                broadcasted_aggregate,
                action,
                method="log_prob",
            )

        return jnp.moveaxis(jax.vmap(_log_prob, in_axes=(1))(actions), 0, -1)

    def dist_prob(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        aggregate_hidden_state: jnp.ndarray,
        aggregate_done: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}

        def fn(broadcasted_aggregate):
            logits = self._action_logits(
                policy_params, individual_states, broadcasted_aggregate
            )
            prob = jax.nn.softmax(logits, axis=-1)
            return prob

        return self._with_aggregate_embedding(
            params,
            individual_states,
            aggregate_obs,
            aggregate_hidden_state,
            aggregate_done,
            fn,
        )


class MeanFieldContinuousPolicy:
    def __init__(
        self,
        state_type: str,
        num_states: int | None = None,
        actions: jnp.ndarray | None = None,
        n_actions_per_dim: int | None = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # --- discretization settings ---
        # Mean Field Sequence Generation expects a matrix of probabilities for each action - i.e. prob_a over discrete actions.
        # Turn the continuous density into categorical logits by evaluating log_prob on `action_grid`, then softmaxing the logits.
        self.state_type = state_type
        self.num_states = num_states
        policy_kwargs = dict(policy_kwargs) if policy_kwargs else {}
        default_policy_kwargs = dict(
            activation="tanh",
            hidden_layer_sizes=(64, 64, 64),
            action_dim=1,
            action_range=(0.0, 1.0),
            state_type=self.state_type,
            num_states=self.num_states,
        )
        default_policy_kwargs.update(policy_kwargs)
        self.policy = BetaPolicy(**default_policy_kwargs)
        self.action_dim = int(default_policy_kwargs["action_dim"])

        if actions is None:
            assert n_actions_per_dim is not None, (
                "n_actions_per_dim must be provided if actions is not provided"
            )
            per_dim_axes = [
                jnp.linspace(
                    default_policy_kwargs["action_range"][0],
                    default_policy_kwargs["action_range"][1],
                    n_actions_per_dim,
                )
                for i in range(self.action_dim)
            ]
            mesh = jnp.meshgrid(*per_dim_axes, indexing="ij")
            self.action_grid = jnp.stack([m.reshape(-1) for m in mesh], axis=-1)
        else:
            if actions.ndim == 1:
                actions = actions.reshape(-1, self.action_dim)
            self.action_grid = actions
            print(f"action_grid shape: {self.action_grid.shape}")

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
        rng: jnp.ndarray,
    ):
        return self.dist_prob(params, individual_states, aggregate_obs)

    def init(
        self,
        rng: jnp.ndarray,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
        rng_action: jnp.ndarray,
    ):
        broadcasted = self._broadcast_aggregate_obs(individual_states, aggregate_obs[0])
        return self.policy.init(rng, individual_states, broadcasted, rng_action)

    def _action_logits(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        broadcasted_aggregate: jnp.ndarray,
    ) -> jnp.ndarray:
        actions = jnp.broadcast_to(
            self.action_grid[None, ...],
            (
                individual_states.shape[0],
                self.action_grid.shape[0],
                self.action_grid.shape[-1],
            ),
        )

        def _log_prob(action):
            return self.policy.apply(
                params,
                individual_states,
                broadcasted_aggregate,
                action,
                method="log_prob",
            )

        return jnp.moveaxis(jax.vmap(_log_prob, in_axes=(1))(actions), 0, -1)

    def dist_prob(
        self,
        params: dict,
        individual_states: jnp.ndarray,
        aggregate_obs: jnp.ndarray,
    ):

        def fn(broadcasted_aggregate):
            logits = self._action_logits(
                params, individual_states, broadcasted_aggregate
            )
            prob = jax.nn.softmax(logits, axis=-1)
            return prob

        return self._with_broadcasted_aggregate(
            params, individual_states, aggregate_obs, fn
        )
