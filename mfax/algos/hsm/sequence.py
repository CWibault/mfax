"""
Generates a mean field sequence from a mean field policy.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from mfax.envs.pushforward.base import PushforwardMFSequence


def mf_sequence(
    rng,
    env,
    agent,
    state_type,
    num_envs,
    max_steps_in_episode,
) -> Tuple[PushforwardMFSequence]:
    """
    Generates a mean field sequence from a (recurrent) mean field policy.
    env: Mean-Field environment - i.e. steps entire Mean Field forward.
    agent: Mean-Field policy. i.e. policy must be wrapped in MFActorWrapper or MFRecurrentActorWrapper.
    """

    use_recurrent = hasattr(agent, "init_hidden")
    if state_type == "indices":
        individual_states = env.state_indices
    else:
        individual_states = env.params.states
    print(f"Using recurrent policy: {use_recurrent}")

    if use_recurrent:

        @jax.jit
        def _select_action(aggregate_obs, hidden_state, done_mask):
            prob_a, next_hidden = agent(
                individual_states, aggregate_obs, hidden_state, done=done_mask
            )
            return prob_a, next_hidden

        @jax.jit
        def _policy_and_env_step(runner_state, _):
            (
                last_aggregate_s,
                last_aggregate_obs,
                last_aggregate_terminated,
                last_aggregate_truncated,
                last_hidden,
                rng,
            ) = runner_state

            # --- select action ---
            done_mask = jnp.logical_or(
                last_aggregate_terminated, last_aggregate_truncated
            )
            prob_a, next_hidden = _select_action(
                last_aggregate_obs, last_hidden, done_mask
            )

            # --- step environment ---
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, num_envs)
            (
                aggregate_obs,
                _,
                aggregate_s,
                _,
                mat_r,
                aggregate_terminated,
                aggregate_truncated,
                _,
            ) = jax.vmap(env.mf_step, in_axes=(0, 0, 0))(
                rng_step, last_aggregate_s, prob_a
            )

            # --- only accumulate rewards if environment is not done ---
            aggregate_terminated = aggregate_terminated | last_aggregate_terminated
            aggregate_truncated = aggregate_truncated | last_aggregate_truncated

            # --- transition ---
            transition = PushforwardMFSequence(
                aggregate_s=last_aggregate_s,
                aggregate_obs=last_aggregate_obs,
                aggregate_hidden=last_hidden,
                prob_a=prob_a,
                mat_r=mat_r,
                aggregate_terminated=aggregate_terminated,
                aggregate_truncated=aggregate_truncated,
            )
            runner_state = (
                aggregate_s,
                aggregate_obs,
                aggregate_terminated,
                aggregate_truncated,
                next_hidden,
                rng,
            )
            return runner_state, transition

    else:

        @jax.jit
        def _select_action(aggregate_obs):
            prob_a = agent(individual_states, aggregate_obs)
            return prob_a

        @jax.jit
        def _policy_and_env_step(runner_state, _):
            (
                last_aggregate_s,
                last_aggregate_obs,
                last_aggregate_terminated,
                last_aggregate_truncated,
                rng,
            ) = runner_state

            # --- select action ---
            prob_a = _select_action(last_aggregate_obs)

            # --- step environment ---
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, num_envs)
            (
                aggregate_obs,
                _,
                aggregate_s,
                _,
                mat_r,
                aggregate_terminated,
                aggregate_truncated,
                _,
            ) = jax.vmap(env.mf_step, in_axes=(0, 0, 0))(
                rng_step, last_aggregate_s, prob_a
            )

            # --- only accumulate rewards if environment is not done ---
            aggregate_terminated = aggregate_terminated | last_aggregate_terminated
            aggregate_truncated = aggregate_truncated | last_aggregate_truncated

            # --- transition ---
            transition = PushforwardMFSequence(
                aggregate_s=last_aggregate_s,
                aggregate_obs=last_aggregate_obs,
                aggregate_hidden=None,
                prob_a=prob_a,
                mat_r=mat_r,
                aggregate_terminated=aggregate_terminated,
                aggregate_truncated=aggregate_truncated,
            )
            runner_state = (
                aggregate_s,
                aggregate_obs,
                aggregate_terminated,
                aggregate_truncated,
                rng,
            )
            return runner_state, transition

    # --- initialise environment ---
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, num_envs)
    init_aggregate_obs, init_aggregate_s = jax.vmap(env.mf_reset, in_axes=(0,))(
        reset_rng
    )
    init_aggregate_terminated = jnp.zeros((num_envs,), dtype=int)
    init_aggregate_truncated = jnp.zeros((num_envs,), dtype=int)

    if use_recurrent:
        init_hidden = agent.init_hidden(num_envs)
        runner_state = (
            init_aggregate_s,
            init_aggregate_obs,
            init_aggregate_terminated,
            init_aggregate_truncated,
            init_hidden,
            rng,
        )
    else:
        runner_state = (
            init_aggregate_s,
            init_aggregate_obs,
            init_aggregate_terminated,
            init_aggregate_truncated,
            rng,
        )

    # --- collect trajectories ---
    _, traj_batch = jax.lax.scan(
        _policy_and_env_step, runner_state, None, int(max_steps_in_episode)
    )

    return traj_batch


def make_mf_sequence(env, agent, num_envs, max_steps_in_episode, state_type):

    use_recurrent = hasattr(agent, "init_hidden")
    if state_type == "indices":
        individual_states = env.state_indices
    else:
        individual_states = env.params.states
    print(f"Using recurrent policy: {use_recurrent}")

    @jax.jit
    def _mf_sequence(
        rng,
        agent_params,
    ) -> Tuple[PushforwardMFSequence]:
        """
        Generates a mean field sequence from a (recurrent) mean field policy.
        env: Mean-Field environment - i.e. steps entire Mean Field forward.
        agent: Mean-Field policy. i.e. policy must be wrapped in MFActorWrapper or MFRecurrentActorWrapper.
        """

        if use_recurrent:

            def _select_action(aggregate_obs, hidden_state, done_mask):
                prob_a, next_hidden = agent(
                    individual_states,
                    aggregate_obs,
                    hidden_state,
                    done=done_mask,
                    mf_params=agent_params,
                )
                return prob_a, next_hidden

            def _policy_and_env_step(runner_state, _):
                (
                    last_aggregate_s,
                    last_aggregate_obs,
                    last_aggregate_terminated,
                    last_aggregate_truncated,
                    last_hidden,
                    rng,
                ) = runner_state

                # --- select action ---
                done_mask = jnp.logical_or(
                    last_aggregate_terminated, last_aggregate_truncated
                )
                prob_a, next_hidden = _select_action(
                    last_aggregate_obs, last_hidden, done_mask
                )

                # --- step environment ---
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                (
                    aggregate_obs,
                    _,
                    aggregate_s,
                    _,
                    mat_r,
                    aggregate_terminated,
                    aggregate_truncated,
                    _,
                ) = jax.vmap(env.mf_step, in_axes=(0, 0, 0))(
                    rng_step, last_aggregate_s, prob_a
                )

                # --- only accumulate rewards if environment is not done ---
                aggregate_terminated = aggregate_terminated | last_aggregate_terminated
                aggregate_truncated = aggregate_truncated | last_aggregate_truncated

                # --- transition ---
                transition = PushforwardMFSequence(
                    aggregate_s=last_aggregate_s,
                    aggregate_obs=last_aggregate_obs,
                    aggregate_hidden=last_hidden,
                    prob_a=prob_a,
                    mat_r=mat_r,
                    aggregate_terminated=aggregate_terminated,
                    aggregate_truncated=aggregate_truncated,
                )
                runner_state = (
                    aggregate_s,
                    aggregate_obs,
                    aggregate_terminated,
                    aggregate_truncated,
                    next_hidden,
                    rng,
                )
                return runner_state, transition

        else:

            def _select_action(aggregate_obs):
                prob_a = agent(individual_states, aggregate_obs, mf_params=agent_params)
                return prob_a

            def _policy_and_env_step(runner_state, _):
                (
                    last_aggregate_s,
                    last_aggregate_obs,
                    last_aggregate_terminated,
                    last_aggregate_truncated,
                    rng,
                ) = runner_state

                # --- select action ---
                prob_a = _select_action(last_aggregate_obs)

                # --- step environment ---
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                (
                    aggregate_obs,
                    _,
                    aggregate_s,
                    _,
                    mat_r,
                    aggregate_terminated,
                    aggregate_truncated,
                    _,
                ) = jax.vmap(env.mf_step, in_axes=(0, 0, 0))(
                    rng_step, last_aggregate_s, prob_a
                )

                # --- only accumulate rewards if environment is not done ---
                aggregate_terminated = aggregate_terminated | last_aggregate_terminated
                aggregate_truncated = aggregate_truncated | last_aggregate_truncated

                # --- transition ---
                transition = PushforwardMFSequence(
                    aggregate_s=last_aggregate_s,
                    aggregate_obs=last_aggregate_obs,
                    aggregate_hidden=None,
                    prob_a=prob_a,
                    mat_r=mat_r,
                    aggregate_terminated=aggregate_terminated,
                    aggregate_truncated=aggregate_truncated,
                )
                runner_state = (
                    aggregate_s,
                    aggregate_obs,
                    aggregate_terminated,
                    aggregate_truncated,
                    rng,
                )
                return runner_state, transition

        # --- initialise environment ---
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        init_aggregate_obs, init_aggregate_s = jax.vmap(env.mf_reset, in_axes=(0,))(
            reset_rng
        )
        init_aggregate_terminated = jnp.zeros((num_envs,), dtype=int)
        init_aggregate_truncated = jnp.zeros((num_envs,), dtype=int)

        if use_recurrent:
            init_hidden = agent.init_hidden(num_envs)
            runner_state = (
                init_aggregate_s,
                init_aggregate_obs,
                init_aggregate_terminated,
                init_aggregate_truncated,
                init_hidden,
                rng,
            )
        else:
            runner_state = (
                init_aggregate_s,
                init_aggregate_obs,
                init_aggregate_terminated,
                init_aggregate_truncated,
                rng,
            )

        # --- collect trajectories ---
        _, traj_batch = jax.lax.scan(
            _policy_and_env_step, runner_state, None, int(max_steps_in_episode)
        )

        return traj_batch

    return _mf_sequence
