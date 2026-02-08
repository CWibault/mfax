import jax
import jax.numpy as jnp
from typing import Tuple

from mfax.envs.pushforward.base import PushforwardMFSequence, PushforwardAggregateState


def br(
    env,
    traj_batch: PushforwardMFSequence,
    gamma: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Evaluates best response to a given mean field and common noise sequence using backward induction.
    """
    assert traj_batch.aggregate_s.mu.ndim == 3, (
        "Trajectory batch must have shape (num_steps, num_envs, num_states)"
    )
    num_envs = traj_batch.aggregate_s.mu.shape[1]
    n_states = traj_batch.aggregate_s.mu.shape[-1]
    n_actions = traj_batch.prob_a.shape[-1]
    state_actions = jnp.broadcast_to(
        jnp.arange(n_actions)[:, None], (n_actions, n_states)
    )

    @jax.jit
    def _compute_best_action(
        mat_r: jnp.ndarray,
        aggregate_s: PushforwardAggregateState,
        future_disc_rewards: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Computes best action for each state at current timestep.
        i.e. evaluates all actions and selects the one that maximizes
        the q function (immediate reward + discounted future value).
        The undiscounted rewards are also returned,
        but the actions are selected using the discounted rewards.
        """

        def _evaluate_action(vec_a: jnp.ndarray):
            """Evaluates a specific action for all states."""

            def _value_per_env(mat_r, aggregate_s, future_disc_rewards, vec_a):
                vec_r = jnp.take_along_axis(mat_r, vec_a[..., None], axis=-1).squeeze(
                    -1
                )
                prob_a = jax.nn.one_hot(vec_a, n_actions, axis=-1)
                q_val = vec_r + gamma * env.mf_expected_value(
                    future_disc_rewards, prob_a, aggregate_s
                )
                return q_val, vec_r

            return jax.vmap(_value_per_env, in_axes=(0, 0, 0, None))(
                mat_r, aggregate_s, future_disc_rewards, vec_a
            )

        q_values, immediate_rewards = jax.vmap(_evaluate_action, in_axes=(0))(
            state_actions
        )
        best_idx = jnp.argmax(
            q_values, axis=0, keepdims=True
        )  # [1, num_envs, n_states]
        best_disc_rewards = jnp.take_along_axis(q_values, best_idx, axis=0).squeeze(
            0
        )  # [num_envs, n_states]
        best_rewards = jnp.take_along_axis(immediate_rewards, best_idx, axis=0).squeeze(
            0
        )  # [num_envs, n_states]
        best_actions = best_idx.squeeze(0)  # [num_envs, n_states]
        return best_disc_rewards, best_actions, best_rewards

    def _backward_induction_scan(
        traj_batch: PushforwardMFSequence,
        init_disc_rewards: jnp.ndarray,
        init_undisc_rewards: jnp.ndarray,
    ):
        def _step(carry, inputs):
            """Processes one timestep in backward induction."""
            mat_r, aggregate_s = inputs
            future_disc_rewards, future_undisc_rewards = carry

            best_disc_rewards, best_actions, best_rewards = _compute_best_action(
                mat_r, aggregate_s, future_disc_rewards
            )
            br_actions_onehot = jax.nn.one_hot(best_actions, n_actions, axis=-1)

            undisc_rewards = best_rewards + jax.vmap(
                env.mf_expected_value, in_axes=(0, 0, 0)
            )(future_undisc_rewards, br_actions_onehot, aggregate_s)

            return (best_disc_rewards, undisc_rewards), (best_actions, best_rewards)

        return jax.lax.scan(
            _step,
            (init_disc_rewards, init_undisc_rewards),
            (traj_batch.mat_r, traj_batch.aggregate_s),
            reverse=True,
        )

    init_disc_rewards = jnp.zeros((num_envs, n_states))
    init_undisc_rewards = jnp.zeros((num_envs, n_states))
    (discounted_rewards, undiscounted_rewards), (best_actions, best_rewards) = (
        _backward_induction_scan(traj_batch, init_disc_rewards, init_undisc_rewards)
    )
    return discounted_rewards, undiscounted_rewards, best_actions, best_rewards
