"""
Wrappers for mean-field policies.
"""

from typing import Any
import jax.numpy as jnp
import flax.linen as nn


# --- agent-wrapper ---
class SAActorWrapper:
    def __init__(
        self,
        policy: nn.Module,
        params: dict,
        obs_normalizer: Any,
        normalize_obs: bool,
        individual_s_normalizer: Any,
        normalize_individual_s: bool,
    ):
        self.policy = policy
        self.params = params
        self.obs_normalizer = obs_normalizer
        self.normalize_obs = normalize_obs
        self.individual_s_normalizer = individual_s_normalizer
        self.normalize_individual_s = normalize_individual_s

    def __call__(self, individual_states, aggregate_obs):
        aggregate_obs = self.obs_normalizer(aggregate_obs, self.normalize_obs)
        individual_states = self.individual_s_normalizer(
            individual_states, self.normalize_individual_s
        )
        action = self.policy.apply(
            self.params, individual_states, aggregate_obs, method="mode"
        )
        return action


class SARecurrentActorWrapper:
    def __init__(
        self,
        policy: nn.Module,
        params: dict,
        obs_normalizer: Any,
        normalize_obs: bool,
        individual_s_normalizer: Any,
        normalize_individual_s: bool,
    ):
        self.policy = policy
        self.params = params
        self.obs_normalizer = obs_normalizer
        self.normalize_obs = normalize_obs
        self.individual_s_normalizer = individual_s_normalizer
        self.normalize_individual_s = normalize_individual_s
        self.hidden_size = policy.encoder.hidden_size

    def init_hidden(self, batch_size: int) -> jnp.ndarray:
        return self.policy.init_hidden(batch_size, self.hidden_size)

    def __call__(self, individual_states, individual_obs, hidden_state, done=None):
        if done is None:
            done = jnp.zeros((individual_obs.shape[0],), dtype=bool)
        individual_obs = self.obs_normalizer(individual_obs, self.normalize_obs)
        individual_states = self.individual_s_normalizer(
            individual_states, self.normalize_individual_s
        )
        action, next_hidden = self.policy.mode(
            self.params, individual_states, individual_obs, hidden_state, done
        )
        return action, next_hidden


class SAQNetWrapper:
    def __init__(
        self,
        qnet: nn.Module,
        params: dict,
        obs_normalizer: Any,
        normalize_obs: bool,
        individual_s_normalizer: Any,
        normalize_individual_s: bool,
    ):
        self.qnet = qnet
        self.params = params
        self.obs_normalizer = obs_normalizer
        self.normalize_obs = normalize_obs
        self.individual_s_normalizer = individual_s_normalizer
        self.normalize_individual_s = normalize_individual_s

    def __call__(self, individual_obs, aggregate_obs):
        aggregate_obs = self.obs_normalizer(aggregate_obs, self.normalize_obs)
        individual_obs = self.individual_s_normalizer(
            individual_obs, self.normalize_individual_s
        )
        q_vals, action = self.qnet.apply(
            self.params, individual_obs, aggregate_obs, method="argmax"
        )
        return action


class SARecurrentQNetWrapper:
    def __init__(
        self,
        qnet: nn.Module,
        params: dict,
        obs_normalizer: Any,
        normalize_obs: bool,
        individual_s_normalizer: Any,
        normalize_individual_s: bool,
    ):
        self.qnet = qnet
        self.params = params
        self.obs_normalizer = obs_normalizer
        self.normalize_obs = normalize_obs
        self.individual_s_normalizer = individual_s_normalizer
        self.normalize_individual_s = normalize_individual_s
        self.hidden_size = qnet.encoder.hidden_size

    def __call__(self, individual_states, aggregate_obs, hidden_state, done=None):
        if done is None:
            done = jnp.zeros((aggregate_obs.shape[0],), dtype=bool)
        aggregate_obs = self.obs_normalizer(aggregate_obs, self.normalize_obs)
        individual_states = self.individual_s_normalizer(
            individual_states, self.normalize_individual_s
        )
        (_, action), next_hidden = self.qnet.argmax(
            self.params, individual_states, aggregate_obs, hidden_state, done
        )
        return action, next_hidden


# --- value-wrapper ---
class SAValueWrapper:
    def __init__(
        self,
        value: nn.Module,
        params: dict,
        obs_normalizer: Any,
        normalize_obs: bool,
        individual_s_normalizer: Any,
        normalize_individual_s: bool,
    ):
        self.value = value
        self.params = params
        self.obs_normalizer = obs_normalizer
        self.normalize_obs = normalize_obs
        self.individual_s_normalizer = individual_s_normalizer
        self.normalize_individual_s = normalize_individual_s

    def __call__(self, individual_states, aggregate_obs):
        aggregate_obs = self.obs_normalizer(aggregate_obs, self.normalize_obs)
        individual_states = self.individual_s_normalizer(
            individual_states, self.normalize_individual_s
        )
        value = self.value.apply(self.params, individual_states, aggregate_obs)
        return value


class SARecurrentValueWrapper:
    def __init__(
        self,
        value: nn.Module,
        params: dict,
        obs_normalizer: Any,
        normalize_obs: bool,
        individual_s_normalizer: Any,
        normalize_individual_s: bool,
    ):
        self.value = value
        self.params = params
        self.obs_normalizer = obs_normalizer
        self.normalize_obs = normalize_obs
        self.individual_s_normalizer = individual_s_normalizer
        self.normalize_individual_s = normalize_individual_s
        self.hidden_size = value.encoder.hidden_size

    def init_hidden(self, batch_size: int) -> jnp.ndarray:
        return self.value.init_hidden(batch_size, self.hidden_size)

    def __call__(self, individual_states, aggregate_obs, hidden_state, done=None):
        if done is None:
            done = jnp.zeros((aggregate_obs.shape[0],), dtype=bool)
        aggregate_obs = self.obs_normalizer(aggregate_obs, self.normalize_obs)
        individual_states = self.individual_s_normalizer(
            individual_states, self.normalize_individual_s
        )
        value = self.value(
            self.params, individual_states, aggregate_obs, hidden_state, done
        )
        return value
