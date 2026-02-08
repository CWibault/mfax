"""
Wrappers for mean-field policies.
"""

from typing import Any
import jax.numpy as jnp

from mfax.algos.hsm.utils.mf_policy_wrappers import (
    MeanFieldPolicy,
    RecurrentMeanFieldPolicy,
)
from mfax.algos.hsm.utils.mf_value_wrappers import (
    MeanFieldValue,
    RecurrentMeanFieldValue,
)
from mfax.algos.hsm.utils.mf_qnet_wrappers import MeanFieldQNet, RecurrentMeanFieldQNet


# --- agent-wrapper ---
class MFActorWrapper:
    def __init__(
        self,
        mf_policy: MeanFieldPolicy,
        mf_params: dict,
        obs_normalizer: Any,
        normalize_obs: bool,
        individual_s_normalizer: Any,
        normalize_individual_s: bool,
    ):
        self.mf_policy = mf_policy
        self.mf_params = mf_params
        self.obs_normalizer = obs_normalizer
        self.normalize_obs = normalize_obs
        self.individual_s_normalizer = individual_s_normalizer
        self.normalize_individual_s = normalize_individual_s

    def __call__(self, individual_states, aggregate_obs, mf_params=None):
        if mf_params is None:
            mf_params = self.mf_params
        aggregate_obs = self.obs_normalizer(aggregate_obs, self.normalize_obs)
        individual_states = self.individual_s_normalizer(
            individual_states, self.normalize_individual_s
        )
        prob_a = self.mf_policy.dist_prob(mf_params, individual_states, aggregate_obs)
        return prob_a


class MFRecurrentActorWrapper:
    def __init__(
        self,
        mf_policy: RecurrentMeanFieldPolicy,
        mf_params: dict,
        obs_normalizer: Any,
        normalize_obs: bool,
        individual_s_normalizer: Any,
        normalize_individual_s: bool,
    ):
        self.mf_policy = mf_policy
        self.mf_params = mf_params
        self.obs_normalizer = obs_normalizer
        self.normalize_obs = normalize_obs
        self.individual_s_normalizer = individual_s_normalizer
        self.normalize_individual_s = normalize_individual_s
        self.hidden_size = mf_policy.aggregate_encoder.hidden_size

    def init_hidden(self, batch_size: int) -> jnp.ndarray:
        return self.mf_policy.init_hidden(batch_size, self.hidden_size)

    def __call__(
        self, individual_states, aggregate_obs, hidden_state, done=None, mf_params=None
    ):
        if mf_params is None:
            mf_params = self.mf_params
        if done is None:
            done = jnp.zeros((aggregate_obs.shape[0],), dtype=bool)
        aggregate_obs = self.obs_normalizer(aggregate_obs, self.normalize_obs)
        individual_states = self.individual_s_normalizer(
            individual_states, self.normalize_individual_s
        )
        prob_a, next_hidden = self.mf_policy.dist_prob(
            mf_params, individual_states, aggregate_obs, hidden_state, done
        )
        return prob_a, next_hidden


class MFQNetWrapper:
    def __init__(
        self,
        mf_qnet: MeanFieldQNet,
        mf_params: dict,
        obs_normalizer: Any,
        normalize_obs: bool,
        individual_s_normalizer: Any,
        normalize_individual_s: bool,
    ):
        self.mf_qnet = mf_qnet
        self.mf_params = mf_params
        self.obs_normalizer = obs_normalizer
        self.normalize_obs = normalize_obs
        self.individual_s_normalizer = individual_s_normalizer
        self.normalize_individual_s = normalize_individual_s

    def __call__(self, individual_obs, aggregate_obs, mf_params=None):
        if mf_params is None:
            mf_params = self.mf_params
        aggregate_obs = self.obs_normalizer(aggregate_obs, self.normalize_obs)
        individual_obs = self.individual_s_normalizer(
            individual_obs, self.normalize_individual_s
        )
        _, prob_a = self.mf_qnet.softmax(mf_params, individual_obs, aggregate_obs)
        return prob_a


class MFRecurrentQNetWrapper:
    def __init__(
        self,
        mf_qnet: RecurrentMeanFieldQNet,
        mf_params: dict,
        obs_normalizer: Any,
        normalize_obs: bool,
        individual_s_normalizer: Any,
        normalize_individual_s: bool,
    ):
        self.mf_qnet = mf_qnet
        self.mf_params = mf_params
        self.obs_normalizer = obs_normalizer
        self.normalize_obs = normalize_obs
        self.individual_s_normalizer = individual_s_normalizer
        self.normalize_individual_s = normalize_individual_s
        self.hidden_size = mf_qnet.aggregate_encoder.hidden_size

    def __call__(self, individual_states, aggregate_obs, hidden_state, done=None):
        if done is None:
            done = jnp.zeros((aggregate_obs.shape[0],), dtype=bool)
        aggregate_obs = self.obs_normalizer(aggregate_obs, self.normalize_obs)
        individual_states = self.individual_s_normalizer(
            individual_states, self.normalize_individual_s
        )
        (_, prob_a), next_hidden = self.mf_qnet.softmax(
            self.mf_params, individual_states, aggregate_obs, hidden_state, done
        )
        return prob_a, next_hidden


# --- value-wrapper ---
class MFValueWrapper:
    def __init__(
        self,
        mf_value: MeanFieldValue,
        mf_params: dict,
        obs_normalizer: Any,
        normalize_obs: bool,
        individual_s_normalizer: Any,
        normalize_individual_s: bool,
    ):
        self.mf_value = mf_value
        self.mf_params = mf_params
        self.obs_normalizer = obs_normalizer
        self.normalize_obs = normalize_obs
        self.individual_s_normalizer = individual_s_normalizer
        self.normalize_individual_s = normalize_individual_s

    def __call__(self, individual_states, aggregate_obs):
        aggregate_obs = self.obs_normalizer(aggregate_obs, self.normalize_obs)
        individual_states = self.individual_s_normalizer(
            individual_states, self.normalize_individual_s
        )
        value = self.mf_value(self.mf_params, individual_states, aggregate_obs)
        return value


class MFRecurrentValueWrapper:
    def __init__(
        self,
        mf_value: RecurrentMeanFieldValue,
        mf_params: dict,
        obs_normalizer: Any,
        normalize_obs: bool,
        individual_s_normalizer: Any,
        normalize_individual_s: bool,
    ):
        self.mf_value = mf_value
        self.mf_params = mf_params
        self.obs_normalizer = obs_normalizer
        self.normalize_obs = normalize_obs
        self.individual_s_normalizer = individual_s_normalizer
        self.normalize_individual_s = normalize_individual_s
        self.hidden_size = mf_value.aggregate_encoder.hidden_size

    def init_hidden(self, batch_size: int) -> jnp.ndarray:
        return self.mf_value.init_hidden(batch_size, self.hidden_size)

    def __call__(self, individual_states, aggregate_obs, hidden_state, done=None):
        if done is None:
            done = jnp.zeros((aggregate_obs.shape[0],), dtype=bool)
        aggregate_obs = self.obs_normalizer(aggregate_obs, self.normalize_obs)
        individual_states = self.individual_s_normalizer(
            individual_states, self.normalize_individual_s
        )
        value = self.mf_value(
            self.mf_params, individual_states, aggregate_obs, hidden_state, done
        )
        return value
