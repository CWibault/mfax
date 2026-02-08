import jax
import jax.numpy as jnp
from typing import Tuple

from mfax.algos.hsm.sequence import mf_sequence
from mfax.envs.pushforward.base import PushforwardMFSequence


def calculate_discounted_rewards(
    env,
    gamma,
    traj_batch: PushforwardMFSequence,
    final_disc_rewards,
    final_undisc_rewards,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], None]:
    """Compute discounted and undiscounted rewards given a trajectory batch."""

    def _get_discounted_rewards(carry, transition):
        disc_rewards, undisc_rewards = carry
        prob_a, mat_r, aggregate_s = (
            transition.prob_a,
            transition.mat_r,
            transition.aggregate_s,
        )
        expected = jax.vmap(env.mf_expected_value, in_axes=(0, 0, 0))

        # Rather than calculating expected return, take greedy action:
        disc_rewards = jnp.sum(prob_a * mat_r, axis=-1) + gamma * expected(
            disc_rewards, prob_a, aggregate_s
        )
        undisc_rewards = jnp.sum(prob_a * mat_r, axis=-1) + expected(
            undisc_rewards, prob_a, aggregate_s
        )
        return (disc_rewards, undisc_rewards), None

    return jax.lax.scan(
        _get_discounted_rewards,
        (final_disc_rewards, final_undisc_rewards),
        traj_batch,
        reverse=True,
    )


def evaluate(
    rng,
    env,
    agent,
    state_type,
    gamma: float = 1.0,
    num_envs: int = 2,
    max_steps_in_episode: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate a (recurrent) mean-field policy.
    """
    num_envs = int(num_envs)

    # --- generate sequence using mf_sequence ---
    traj_batch = mf_sequence(
        rng, env, agent, state_type, num_envs, max_steps_in_episode
    )

    # --- calculate discounted rewards ---
    init_final_disc_rewards = jnp.zeros((num_envs, env.n_states))
    init_final_undisc_rewards = jnp.zeros((num_envs, env.n_states))
    (discounted_rewards, undiscounted_rewards), _ = calculate_discounted_rewards(
        env, gamma, traj_batch, init_final_disc_rewards, init_final_undisc_rewards
    )

    return discounted_rewards, undiscounted_rewards


def evaluate_given_sequence(
    env,
    traj_batch: PushforwardMFSequence,
    gamma: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate a given (recurrent) mean-field policy.
    """
    assert traj_batch.aggregate_s.mu.ndim == 3, (
        "Trajectory batch must have shape (num_steps, num_envs, num_states)"
    )
    num_envs = int(traj_batch.aggregate_s.mu.shape[1])

    # --- calculate discounted rewards ---
    init_final_disc_rewards = jnp.zeros((num_envs, env.n_states))
    init_final_undisc_rewards = jnp.zeros((num_envs, env.n_states))
    (discounted_rewards, undiscounted_rewards), _ = calculate_discounted_rewards(
        env, gamma, traj_batch, init_final_disc_rewards, init_final_undisc_rewards
    )

    return discounted_rewards, undiscounted_rewards
