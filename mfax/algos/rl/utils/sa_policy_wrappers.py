"""
Wrappers for single-agent policies.
Wrappers are designed so that the returned parameter dict has the same structure as
the RecurrentMeanFieldPolicy wrapper in `mfax/algos/hsm/utils/mf_policy_wrappers.py`.
"""

import jax.numpy as jnp
from typing import Optional, Dict, Any
import jax

from mfax.utils.nets.policy import DiscretePolicy, BetaPolicy
from mfax.utils.nets.base import RecurrentEncoder


class RecurrentSingleAgentPolicy:
    """
    Single-agent policy wrapper that prepends a "RecurrentEncoder" to a policy.
    obs_t -> RecurrentEncoder(hidden_{t-1}, obs_t, done_t) -> embedding_t
    (state_t, embedding_t) -> Policy -> action_t (+ extras)
    The returned params dict is {"encoder": ..., "policy": ...} so that it is
    compatible with the "RecurrentMeanFieldPolicy" wrapper.
    If "policy_type" == "beta", actions are continuous (direct output of
    "BetaPolicy").
    """

    def __init__(
        self,
        state_type: str,
        num_states: int | None = None,
        policy_type: str = "discrete",  # "discrete" | "beta"
        policy_kwargs: Optional[Dict[str, Any]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.state_type = state_type  # "states" or "indices"
        self.num_states = num_states

        policy_kwargs = dict(policy_kwargs) if policy_kwargs else {}

        default_encoder_kwargs = dict(hidden_size=64, embed_size=64, activation="tanh")
        if encoder_kwargs:
            default_encoder_kwargs.update(encoder_kwargs)
        self.encoder = RecurrentEncoder(**default_encoder_kwargs)

        if policy_type == "discrete":
            default_policy_kwargs = dict(
                activation="tanh",
                hidden_layer_sizes=(64, 64, 64),
                n_actions=1,
                state_type=self.state_type,
                num_states=self.num_states,
            )
            default_policy_kwargs.update(policy_kwargs)
            self.policy = DiscretePolicy(**default_policy_kwargs)
            self.policy_type = "discrete"
        else:
            assert policy_type == "beta", (
                f"Invalid policy_type: {policy_type}. Expected 'discrete' or 'beta'."
            )
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
            self.policy_type = "beta"

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
        rng: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}
        new_hidden, obs_embedding = self._encode_obs(
            params, individual_obs, individual_hidden_state, done
        )
        out = self.policy.apply(policy_params, individual_s, obs_embedding, rng)
        return out, new_hidden

    def init(
        self,
        rng: jnp.ndarray,
        individual_s: jnp.ndarray,
        individual_obs: jnp.ndarray,
        individual_hidden_state: jnp.ndarray | None,
        done: jnp.ndarray,
        rng_action: jnp.ndarray,
    ):
        if individual_hidden_state is None:
            individual_hidden_state = self.init_hidden(
                individual_s.shape[0], self.encoder.hidden_size
            )
        rng_enc, rng_pol = jax.random.split(rng)
        encoder_params = self.encoder.init(
            rng_enc, individual_hidden_state, individual_obs, done
        )["params"]

        _, init_embeddings = self.encoder.apply(
            {"params": encoder_params},
            individual_hidden_state,
            individual_obs,
            done,
        )
        policy_params = self.policy.init(
            rng_pol, individual_s, init_embeddings, rng_action
        )["params"]
        return {"encoder": encoder_params, "policy": policy_params}

    def _action_dist(
        self,
        params: dict,
        individual_s: jnp.ndarray,
        individual_obs: jnp.ndarray,
        individual_hidden_state: jnp.ndarray,
        done: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}
        new_hidden, obs_embedding = self._encode_obs(
            params, individual_obs, individual_hidden_state, done
        )
        out = self.policy.apply(
            policy_params, individual_s, obs_embedding, method="_action_dist"
        )
        return out, new_hidden

    def sample(
        self,
        params: dict,
        individual_s: jnp.ndarray,
        individual_obs: jnp.ndarray,
        individual_hidden_state: jnp.ndarray,
        done: jnp.ndarray,
        rng: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}
        new_hidden, obs_embedding = self._encode_obs(
            params, individual_obs, individual_hidden_state, done
        )
        out = self.policy.apply(
            policy_params, individual_s, obs_embedding, rng, method="sample"
        )
        return out, new_hidden

    def sample_and_log_prob(
        self,
        params: dict,
        individual_s: jnp.ndarray,
        individual_obs: jnp.ndarray,
        individual_hidden_state: jnp.ndarray,
        done: jnp.ndarray,
        rng: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}
        new_hidden, obs_embedding = self._encode_obs(
            params, individual_obs, individual_hidden_state, done
        )
        out = self.policy.apply(
            policy_params,
            individual_s,
            obs_embedding,
            rng,
            method="sample_and_log_prob",
        )
        return out, new_hidden

    def mode(
        self,
        params: dict,
        individual_s: jnp.ndarray,
        individual_obs: jnp.ndarray,
        individual_hidden_state: jnp.ndarray,
        done: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}
        new_hidden, obs_embedding = self._encode_obs(
            params, individual_obs, individual_hidden_state, done
        )
        out = self.policy.apply(
            policy_params, individual_s, obs_embedding, method="mode"
        )
        return out, new_hidden

    def log_prob(
        self,
        params: dict,
        individual_s: jnp.ndarray,
        individual_obs: jnp.ndarray,
        individual_hidden_state: jnp.ndarray,
        done: jnp.ndarray,
        actions: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}
        new_hidden, obs_embedding = self._encode_obs(
            params, individual_obs, individual_hidden_state, done
        )
        out = self.policy.apply(
            policy_params, individual_s, obs_embedding, actions, method="log_prob"
        )
        return out, new_hidden

    def entropy(
        self,
        params: dict,
        individual_s: jnp.ndarray,
        individual_obs: jnp.ndarray,
        individual_hidden_state: jnp.ndarray,
        done: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}
        new_hidden, obs_embedding = self._encode_obs(
            params, individual_obs, individual_hidden_state, done
        )
        out = self.policy.apply(
            policy_params, individual_s, obs_embedding, method="entropy"
        )
        return out, new_hidden

    def log_prob_entropy(
        self,
        params: dict,
        individual_s: jnp.ndarray,
        individual_obs: jnp.ndarray,
        individual_hidden_state: jnp.ndarray,
        done: jnp.ndarray,
        actions: jnp.ndarray,
    ):
        policy_params = {"params": params["policy"]}
        new_hidden, obs_embedding = self._encode_obs(
            params, individual_obs, individual_hidden_state, done
        )
        out = self.policy.apply(
            policy_params,
            individual_s,
            obs_embedding,
            actions,
            method="log_prob_entropy",
        )
        return out, new_hidden
