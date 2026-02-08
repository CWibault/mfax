from typing import Sequence, Callable

import jax
import jax.numpy as jnp

import distrax
from flax import linen as nn

from mfax.utils.nets.base import MLP


class DiscretePolicy(nn.Module):
    """
    Categorical policy for discrete actions.
    """

    n_actions: int
    hidden_layer_sizes: Sequence[int]
    activation: Callable
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
        self.action_logits = nn.Dense(
            self.n_actions,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.normal(stddev=1e-2),
        )

    def __call__(self, state, obs, rng):
        if self.state_type == "states":
            assert obs.ndim == state.ndim, (
                f"obs.ndim ({obs.ndim}) must equal state.ndim ({state.ndim})"
            )
        else:
            assert obs.ndim == state.ndim + 1, (
                f"obs.ndim ({obs.ndim}) must be one more than state.ndim ({state.ndim})"
            )
        action_dist = self._action_dist(state, obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action)

    def _action_dist(self, state, obs):
        if self.state_type == "states":
            assert obs.ndim == state.ndim, (
                f"obs.ndim ({obs.ndim}) must equal state.ndim ({state.ndim})"
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
        action_logits = self.action_logits(features)
        return distrax.Categorical(logits=action_logits)

    def sample_and_log_prob(self, state, obs, rng):
        return self(state, obs, rng)

    def sample(self, state, obs, rng):
        action, _ = self(state, obs, rng)
        return action

    def mode(self, state, obs):
        action_dist = self._action_dist(state, obs)
        action = action_dist.mode()
        return action

    def log_prob(self, state, obs, action):
        action_dist = self._action_dist(state, obs)
        return action_dist.log_prob(action)

    def entropy(self, state, obs):
        action_dist = self._action_dist(state, obs)
        return action_dist.entropy()

    def log_prob_entropy(self, state, obs, action):
        action_dist = self._action_dist(state, obs)
        return action_dist.log_prob(action), action_dist.entropy()

    def dist_prob(self, state, obs):
        action_dist = self._action_dist(state, obs)
        return jax.nn.softmax(action_dist.logits, axis=-1)

    def dist_prob_sample_and_log_prob(self, state, obs, rng):
        action_dist = self._action_dist(state, obs)
        prob = jax.nn.softmax(action_dist.logits, axis=-1)
        action = action_dist.sample(seed=rng)
        return prob, action, action_dist.log_prob(action)

    def dist_log_prob_entropy(self, state, obs):
        action_dist = self._action_dist(state, obs)
        return jax.nn.log_softmax(action_dist.logits, axis=-1), action_dist.entropy()


class BetaPolicy(nn.Module):
    """
    Beta policy for continuous actions.
    """

    action_dim: int
    action_range: tuple[float, float]
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    state_type: str
    num_states: int | None = None

    @property
    def action_loc(self):
        return self.action_range[0]

    @property
    def action_scale(self):
        return self.action_range[1] - self.action_range[0]

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
        self.alpha = nn.Dense(self.action_dim)
        self.beta = nn.Dense(self.action_dim)

    def __call__(self, state, obs, rng):
        if self.state_type == "states":
            assert obs.ndim == state.ndim, (
                f"obs.ndim ({obs.ndim}) must equal state.ndim ({state.ndim})"
            )
        else:
            assert obs.ndim == state.ndim + 1, (
                f"obs.ndim ({obs.ndim}) must be one more than state.ndim ({state.ndim})"
            )
        action_dist = self._action_dist(state, obs)
        action = jnp.clip(action_dist.sample(seed=rng), 1e-6, 1.0 - 1e-6)
        log_prob = action_dist.log_prob(action)
        action = self.action_loc + action * self.action_scale
        return action, log_prob.sum(axis=-1)

    def _action_dist(self, state, obs):
        if self.state_type == "states":
            assert obs.ndim == state.ndim, (
                f"obs.ndim ({obs.shape}) must equal state.ndim ({state.shape})"
            )
        else:
            assert obs.ndim == state.ndim + 1, (
                f"obs.ndim ({obs.shape}) must be one more than state.ndim ({state.shape})"
            )
        state_embedding = self.state_embedding(state)
        obs_embedding = self.obs_embedding(obs)
        features = self.features(
            jnp.concatenate([state_embedding, obs_embedding], axis=-1)
        )
        alpha = 1 + nn.softplus(self.alpha(features)) + 1e-6
        beta = 1 + nn.softplus(self.beta(features)) + 1e-6
        return distrax.Beta(alpha, beta)

    def sample_and_log_prob(self, state, obs, rng):
        return self(state, obs, rng)

    def sample(self, state, obs, rng):
        action, _ = self(state, obs, rng)
        return action

    def mode(self, state, obs):
        action_dist = self._action_dist(state, obs)
        action = action_dist.mode()
        action = self.action_loc + action * self.action_scale
        return action

    def log_prob(self, state, obs, action):
        action_dist = self._action_dist(state, obs)
        action = jnp.clip(
            (action - self.action_loc) / self.action_scale, 1e-6, 1.0 - 1e-6
        )  # clip action between 1e-6 and 1.0 - 1e-6 to avoid numerical issues
        return action_dist.log_prob(action).sum(axis=-1)

    def entropy(self, state, obs):
        action_dist = self._action_dist(state, obs)
        return action_dist.entropy()

    def log_prob_entropy(self, state, obs, action):
        action_dist = self._action_dist(state, obs)
        action = jnp.clip(
            (action - self.action_loc) / self.action_scale, 1e-6, 1.0 - 1e-6
        )
        log_prob = action_dist.log_prob(action)
        return log_prob.sum(axis=-1), action_dist.entropy()


class GaussianPolicy(nn.Module):
    """
    Gaussian policy for continuous actions.
    """

    action_dim: int
    action_range: tuple[float, float]
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    log_std_range: tuple[float, float]
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
        self.action_mean = nn.Dense(self.action_dim)
        self.action_log_std = nn.Dense(self.action_dim)

    def __call__(self, state, obs, rng):
        if self.state_type == "states":
            assert obs.ndim == state.ndim, (
                f"obs.ndim ({obs.ndim}) must equal state.ndim ({state.ndim})"
            )
        else:
            assert obs.ndim == state.ndim + 1, (
                f"obs.ndim ({obs.ndim}) must be one more than state.ndim ({state.ndim})"
            )
        action_dist = self._action_dist(state, obs)
        action = action_dist.sample(seed=rng)
        log_prob = action_dist.log_prob(action)
        return action, log_prob

    def _action_dist(self, state, obs):
        if self.state_type == "states":
            assert obs.ndim == state.ndim, (
                f"obs.ndim ({obs.ndim}) must equal state.ndim ({state.ndim})"
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
        action_mean = self.action_mean(features)
        action_log_std = self.action_log_std(features)
        action_log_std = jnp.clip(action_log_std, *self.log_std_range)
        return distrax.MultivariateNormalDiag(
            loc=action_mean, scale_diag=jnp.exp(action_log_std)
        )

    def sample_and_log_prob(self, state, obs, rng):
        return self(state, obs, rng)

    def sample(self, state, obs, rng):
        action, _ = self(state, obs, rng)
        return action

    def mode(self, state, obs):
        action_dist = self._action_dist(state, obs)
        action = action_dist.mode()
        return action

    def log_prob(self, state, obs, action):
        action_dist = self._action_dist(state, obs)
        # distrax.MultivariateNormalDiag.log_prob already returns a scalar per sample
        # (i.e. integrates over event dimension), so do not sum again.
        return action_dist.log_prob(action)

    def entropy(self, state, obs):
        action_dist = self._action_dist(state, obs)
        return action_dist.entropy()

    def log_prob_entropy(self, state, obs, action):
        action_dist = self._action_dist(state, obs)
        log_prob = action_dist.log_prob(action)
        return log_prob, action_dist.entropy()
