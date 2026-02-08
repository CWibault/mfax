"""
Generates a mean field sequence from a mean field policy.
"""

import jax
import jax.numpy as jnp

from mfax.envs.sample.base import SampleMFSequence


def mf_sequence(
    rng,
    env,
    agent,
    num_envs,
    max_steps_in_episode,
) -> SampleMFSequence:
    """
    Generates a mean field sequence from a (recurrent) mean field policy.
    env: Mean-Field environment - i.e. steps entire Mean Field forward.
    agent: Mean-Field policy. i.e. policy must be wrapped in MFActorWrapper or MFRecurrentActorWrapper.
    """

    use_recurrent = hasattr(agent, "init_hidden")
    print(f"Using recurrent policy: {use_recurrent}")

    if use_recurrent:

        @jax.jit
        def _select_action(individual_s, individual_obs, hidden_state, done_mask):
            vec_a, next_hidden = agent(
                individual_s, individual_obs, hidden_state, done=done_mask
            )
            return vec_a, next_hidden

        @jax.jit
        def _policy_and_env_step(runner_state, _):
            (
                last_vec_individual_s,
                last_vec_individual_obs,
                last_aggregate_s,
                last_aggregate_terminated,
                last_aggregate_truncated,
                last_actor_hidden,
                rng,
            ) = runner_state

            # --- select action ---
            last_done = jnp.logical_or(
                last_aggregate_terminated, last_aggregate_truncated
            )
            vec_a, next_actor_hidden = jax.vmap(_select_action, in_axes=(0, 0, 0, 0))(
                last_vec_individual_s.state,
                last_vec_individual_obs,
                last_actor_hidden,
                last_done,
            )

            # --- step environment ---
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, num_envs)
            (
                vec_individual_obs,
                _,
                vec_individual_s,
                _,
                aggregate_s,
                _,
                vec_r,
                aggregate_terminated,
                aggregate_truncated,
                _,
            ) = jax.vmap(env.mf_step, in_axes=(0, 0, 0, 0))(
                rng_step, last_vec_individual_s, last_aggregate_s, vec_a
            )

            # --- only accumulate rewards if environment is not done ---
            aggregate_terminated = aggregate_terminated | last_aggregate_terminated
            aggregate_truncated = aggregate_truncated | last_aggregate_truncated

            # --- transition ---
            transition = SampleMFSequence(
                aggregate_s=last_aggregate_s,
                aggregate_terminated=last_aggregate_terminated,
                aggregate_truncated=last_aggregate_truncated,
                vec_a=vec_a,
                vec_r=vec_r,
            )
            runner_state = (
                vec_individual_s,
                vec_individual_obs,
                aggregate_s,
                aggregate_terminated,
                aggregate_truncated,
                next_actor_hidden,
                rng,
            )
            return runner_state, transition

    else:

        @jax.jit
        def _select_action(individual_s, individual_obs):
            vec_a = agent(individual_s, individual_obs)
            return vec_a

        @jax.jit
        def _policy_and_env_step(runner_state, _):
            (
                last_vec_individual_s,
                last_vec_individual_obs,
                last_aggregate_s,
                last_aggregate_terminated,
                last_aggregate_truncated,
                rng,
            ) = runner_state

            # --- select action ---
            vec_a = jax.vmap(_select_action, in_axes=(0, 0))(
                last_vec_individual_s.state, last_vec_individual_obs
            )

            # --- step environment ---
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, num_envs)
            (
                vec_individual_obs,
                _,
                vec_individual_s,
                _,
                aggregate_s,
                _,
                vec_r,
                aggregate_terminated,
                aggregate_truncated,
                _,
            ) = jax.vmap(env.mf_step, in_axes=(0, 0, 0, 0))(
                rng_step, last_vec_individual_s, last_aggregate_s, vec_a
            )

            # --- only accumulate rewards if environment is not done ---
            aggregate_terminated = aggregate_terminated | last_aggregate_terminated
            aggregate_truncated = aggregate_truncated | last_aggregate_truncated

            # --- transition ---
            transition = SampleMFSequence(
                aggregate_s=last_aggregate_s,
                aggregate_terminated=last_aggregate_terminated,
                aggregate_truncated=last_aggregate_truncated,
                vec_a=vec_a,
                vec_r=vec_r,
            )
            runner_state = (
                vec_individual_s,
                vec_individual_obs,
                aggregate_s,
                aggregate_terminated,
                aggregate_truncated,
                rng,
            )
            return runner_state, transition

    # --- initialise environment ---
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, num_envs)
    init_vec_individual_obs, init_vec_individual_s, init_aggregate_s = jax.vmap(
        env.mf_reset, in_axes=(0,)
    )(reset_rng)
    init_aggregate_terminated = jnp.zeros((num_envs,), dtype=int)
    init_aggregate_truncated = jnp.zeros((num_envs,), dtype=int)

    if use_recurrent:
        init_actor_hidden = agent.init_hidden(num_envs * env.params.n_agents).reshape(
            (num_envs, env.params.n_agents, -1)
        )
        runner_state = (
            init_vec_individual_s,
            init_vec_individual_obs,
            init_aggregate_s,
            init_aggregate_terminated,
            init_aggregate_truncated,
            init_actor_hidden,
            rng,
        )
    else:
        runner_state = (
            init_vec_individual_s,
            init_vec_individual_obs,
            init_aggregate_s,
            init_aggregate_terminated,
            init_aggregate_truncated,
            rng,
        )

    # --- collect trajectories ---
    _, traj_batch = jax.lax.scan(
        _policy_and_env_step, runner_state, None, int(max_steps_in_episode)
    )

    return traj_batch


def make_mf_sequence(env, agent, num_envs, max_steps_in_episode):

    use_recurrent = hasattr(agent, "init_hidden")
    print(f"Using recurrent policy: {use_recurrent}")

    @jax.jit
    def _mf_sequence(
        rng,
        agent_params,
    ) -> SampleMFSequence:
        """
        Generates a mean field sequence from a (recurrent) mean field policy.
        env: Mean-Field environment - i.e. steps entire Mean Field forward.
        agent: Mean-Field policy. i.e. policy must be wrapped in MFActorWrapper or MFRecurrentActorWrapper.
        """

        if use_recurrent:

            def _select_action(individual_s, individual_obs, hidden_state, done_mask):
                vec_a, next_hidden = agent(
                    individual_s,
                    individual_obs,
                    hidden_state,
                    done=done_mask,
                    mf_params=agent_params,
                )
                return vec_a, next_hidden

            def _policy_and_env_step(runner_state, _):
                (
                    last_vec_individual_s,
                    last_vec_individual_obs,
                    last_aggregate_s,
                    last_aggregate_terminated,
                    last_aggregate_truncated,
                    last_actor_hidden,
                    rng,
                ) = runner_state

                # --- select action ---
                last_done = jnp.logical_or(
                    last_aggregate_terminated, last_aggregate_truncated
                )
                vec_a, next_actor_hidden = jax.vmap(
                    _select_action, in_axes=(0, 0, 0, 0)
                )(
                    last_vec_individual_s.state,
                    last_vec_individual_obs,
                    last_actor_hidden,
                    last_done,
                )

                # --- step environment ---
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                (
                    vec_individual_obs,
                    _,
                    vec_individual_s,
                    _,
                    aggregate_s,
                    _,
                    vec_r,
                    aggregate_terminated,
                    aggregate_truncated,
                    _,
                ) = jax.vmap(env.mf_step, in_axes=(0, 0, 0, 0))(
                    rng_step, last_vec_individual_s, last_aggregate_s, vec_a
                )

                # --- only accumulate rewards if environment is not done ---
                aggregate_terminated = aggregate_terminated | last_aggregate_terminated
                aggregate_truncated = aggregate_truncated | last_aggregate_truncated

                # --- transition ---
                transition = SampleMFSequence(
                    aggregate_s=last_aggregate_s,
                    aggregate_terminated=last_aggregate_terminated,
                    aggregate_truncated=last_aggregate_truncated,
                    vec_a=vec_a,
                    vec_r=vec_r,
                )
                runner_state = (
                    vec_individual_s,
                    vec_individual_obs,
                    aggregate_s,
                    aggregate_terminated,
                    aggregate_truncated,
                    next_actor_hidden,
                    rng,
                )
                return runner_state, transition

        else:

            def _select_action(individual_s, individual_obs):
                vec_a = agent(individual_s, individual_obs, mf_params=agent_params)
                return vec_a

            def _policy_and_env_step(runner_state, _):
                (
                    last_vec_individual_s,
                    last_vec_individual_obs,
                    last_aggregate_s,
                    last_aggregate_terminated,
                    last_aggregate_truncated,
                    rng,
                ) = runner_state

                # --- select action ---
                vec_a = jax.vmap(_select_action, in_axes=(0, 0))(
                    last_vec_individual_s.state, last_vec_individual_obs
                )

                # --- step environment ---
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                (
                    vec_individual_obs,
                    _,
                    vec_individual_s,
                    _,
                    aggregate_s,
                    _,
                    vec_r,
                    aggregate_terminated,
                    aggregate_truncated,
                    _,
                ) = jax.vmap(env.mf_step, in_axes=(0, 0, 0, 0))(
                    rng_step, last_vec_individual_s, last_aggregate_s, vec_a
                )

                # --- only accumulate rewards if environment is not done ---
                aggregate_terminated = aggregate_terminated | last_aggregate_terminated
                aggregate_truncated = aggregate_truncated | last_aggregate_truncated

                # --- transition ---
                transition = SampleMFSequence(
                    aggregate_s=last_aggregate_s,
                    aggregate_terminated=last_aggregate_terminated,
                    aggregate_truncated=last_aggregate_truncated,
                    vec_a=vec_a,
                    vec_r=vec_r,
                )
                runner_state = (
                    vec_individual_s,
                    vec_individual_obs,
                    aggregate_s,
                    aggregate_terminated,
                    aggregate_truncated,
                    rng,
                )
                return runner_state, transition

        # --- initialise environment ---
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        init_vec_individual_obs, init_vec_individual_s, init_aggregate_s = jax.vmap(
            env.mf_reset, in_axes=(0,)
        )(reset_rng)
        init_aggregate_terminated = jnp.zeros((num_envs,), dtype=int)
        init_aggregate_truncated = jnp.zeros((num_envs,), dtype=int)

        if use_recurrent:
            init_actor_hidden = agent.init_hidden(
                num_envs * env.params.n_agents
            ).reshape((num_envs, env.params.n_agents, -1))
            runner_state = (
                init_vec_individual_s,
                init_vec_individual_obs,
                init_aggregate_s,
                init_aggregate_terminated,
                init_aggregate_truncated,
                init_actor_hidden,
                rng,
            )
        else:
            runner_state = (
                init_vec_individual_s,
                init_vec_individual_obs,
                init_aggregate_s,
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
