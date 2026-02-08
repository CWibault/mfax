"""Munchausen DQN Agent and deep online mirror descent implementation.
Reference: https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/mfg/examples/mfg_munchausen_domd_jax.py
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
import time
import tyro

import chex
import flashbax as fbx
import flax
from flax.training.train_state import TrainState as BaseTrainState
import jax
import jax.numpy as jnp
import optax
import rlax
import wandb

import sys
from pathlib import Path

# --- make repo root importable even when file is executed by path ---
_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "mfax").is_dir()), None
)
if _REPO_ROOT is not None:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"
os.environ.pop("LD_LIBRARY_PATH", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

from mfax.algos.rl.sequence import mf_sequence  # noqa: E402
from mfax.algos.rl.utils.make_act import SAQNetWrapper  # noqa: E402
from mfax.envs import make_env  # noqa: E402
from mfax.utils.nets.qnet import DiscreteQNet  # noqa: E402
from utils import wandb_log_info, save_pkl  # noqa: E402

# --- for evaluation only, training uses sampled-based mean-field ---
from mfax.algos.hsm.utils.mf_qnet_wrappers import MeanFieldQNet  # noqa: E402
from mfax.algos.hsm.utils.make_act import MFQNetWrapper  # noqa: E402
from mfax.algos.hsm.exploitability import make_exploitability  # noqa: E402

MIN_ACTION_PROB = 1e-6


@dataclass
class args:
    # --- logging ---
    debug: bool = False
    evaluate: bool = True
    log: bool = False
    save: bool = True
    wandb_project: str = "mfax"
    wandb_team: str = ""
    wandb_group: str = "mfax"

    # --- environment and offline dataset ---
    task: str = "endogenous"  # "beach_bar_1d"
    state_type: str = "states"
    discount_factor: float = 0.95
    normalize_obs: bool = True
    normalize_states: bool = True
    common_noise: bool = True

    # --- Munchausen-OMD hyperparameters ---
    algo: str = "rl_m_omd"
    q_net_type: str = "discrete"
    seed: int = 42

    # --- Number of environments for which mean-field sequences are generated.
    # Increasing reduces variance due to common noise or initial distributions, as well as decreasing variance due to transition dynamics.
    num_envs: int = 8  # 128
    # --- Number of agents for stepped forward per mean-field sequence.
    # Increasing reduces variance due to transition dynamics only, hence favour increasing num_envs over num_agents_per_env.
    num_agents_per_env: int = 128  # 8
    # --- Sample batch size.
    # Note that on each "step", num_envs * num_agents_per_env transitions are added to the replay buffer.
    batch_size: int = 2048
    # --- Replay buffer capacity.
    # Should be > num_envs * num_agents_per_env * max_steps_in_episode to contain full trajectories.
    # Should not be larger than num_envs * num_agents_per_env * num_steps_per_iteration, otherwise remains stale between iterations.
    replay_buffer_capacity: int = 300000
    min_buffer_to_learn: int = 10000
    # --- Number of steps after which network is updated.
    learn_every: int = 8  # 64
    # --- Exploration parameter
    # Decay duration percentage in terms of number of total steps.
    epsilon_decay_duration_pct: float = 0.5
    epsilon_power: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    # --- Set to true to prevent stale iterations in replay buffer.
    reset_replay_buffer_on_update: bool = True
    activation: str = "relu"
    # --- Number of steps after which to update target network.
    # num_steps_per_iteration / update_target_every = num_target_updates_per_iteration
    update_target_every: int = 200

    # --- Munchausen parameters ---
    tau: float = 0.05
    alpha: float = 0.99  # 0.95
    with_munchausen: bool = True

    # --- Loss parameters ---
    lr: float = 0.01  # 0.001
    anneal_lr: bool = True
    max_grad_norm: float = 1.0
    loss: str = "mse"
    huber_loss_parameter: float = 1.0

    # --- Iterations ---
    num_iterations: int = 2500  # 200
    num_updates_per_iteration: int = 50  # 200
    eval_frequency: int = 100  # 20


@chex.dataclass(frozen=True)
class Transition:
    individual_obs: chex.Array
    individual_s: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


class TrainState(BaseTrainState):
    prev_q_net_params: flax.core.FrozenDict
    target_q_net_params: flax.core.FrozenDict
    steps: int
    n_updates: int


def make_train_step(args, env, q_net):

    def eps_greedy_exploration(steps):
        return jnp.clip(
            ((args.epsilon_end - args.epsilon_start) / args.epsilon_decay_steps) * steps
            + args.epsilon_start,
            args.epsilon_end,
            args.epsilon_start,
        )

    def _train_step(iteration_runner_state, unused):

        # --- update network ---
        def _update_step(runner_state, unused):

            # --- step environment ---
            (
                ts,
                buffer_state,
                env_mf_sequence,
                last_individual_s,
                last_individual_obs,
                rng,
            ) = runner_state
            rng, _rng = jax.random.split(rng, 2)
            rng_env = jax.random.split(_rng, args.num_envs)

            def _agent_step(
                last_individual_s, last_individual_obs, env_mf_sequence, rng
            ):
                rng, _rng = jax.random.split(rng)
                action_idx = q_net.apply(
                    ts.params,
                    env.normalize_individual_s(
                        last_individual_s.state, args.normalize_states
                    ),
                    env.normalize_obs(last_individual_obs, args.normalize_obs),
                    eps_greedy_exploration(ts.steps),
                    _rng,
                    method="epsilon_greedy",
                )

                # --- step environment ---
                rng_step_agent = jax.random.split(rng, args.num_agents_per_env)
                (
                    individual_obs,
                    individual_s,
                    reward,
                    aggregate_terminated,
                    aggregate_truncated,
                ) = jax.vmap(env.sa_step, in_axes=(0, None, 0, 0))(
                    rng_step_agent, env_mf_sequence, last_individual_s, action_idx
                )
                return (
                    individual_obs,
                    individual_s,
                    reward,
                    aggregate_terminated,
                    aggregate_truncated,
                    action_idx,
                )

            (
                individual_obs,
                individual_s,
                reward,
                aggregate_terminated,
                aggregate_truncated,
                action_idx,
            ) = jax.vmap(_agent_step, in_axes=(0, 0, 1, 0))(
                last_individual_s, last_individual_obs, env_mf_sequence, rng_env
            )
            ts = ts.replace(steps=ts.steps + 1)

            buffer_state = buffer.add(
                buffer_state,
                Transition(
                    individual_obs=last_individual_obs.reshape(
                        args.num_envs * args.num_agents_per_env,
                        *last_individual_obs.shape[2:],
                    ),
                    individual_s=last_individual_s.state.reshape(
                        args.num_envs * args.num_agents_per_env,
                        *last_individual_s.state.shape[2:],
                    ),
                    action=action_idx.reshape(
                        args.num_envs * args.num_agents_per_env,
                        *action_idx.shape[2:],
                    ),
                    reward=reward.reshape(
                        args.num_envs * args.num_agents_per_env,
                        *reward.shape[2:],
                    ),
                    done=jnp.logical_or(
                        aggregate_terminated, aggregate_truncated
                    ).reshape(
                        args.num_envs * args.num_agents_per_env,
                        *aggregate_terminated.shape[2:],
                    ),
                ),
            )

            # --- update q network ---
            def _update_q_net(ts, rng):
                learn_batch = buffer.sample(buffer_state, rng).experience

                def _loss_fn(params):
                    q_vals = q_net.apply(
                        params,
                        env.normalize_individual_s(
                            learn_batch.first.individual_s, args.normalize_states
                        ),
                        env.normalize_obs(
                            learn_batch.first.individual_obs, args.normalize_obs
                        ),
                    )
                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals,
                        jnp.expand_dims(learn_batch.first.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)
                    r_term = learn_batch.first.reward

                    q_next_target_vals = q_net.apply(
                        ts.target_q_net_params,
                        env.normalize_individual_s(
                            learn_batch.second.individual_s, args.normalize_states
                        ),
                        env.normalize_obs(
                            learn_batch.second.individual_obs, args.normalize_obs
                        ),
                    )
                    if args.with_munchausen:
                        _, prob_a = q_net.apply(
                            ts.prev_q_net_params,
                            env.normalize_individual_s(
                                learn_batch.first.individual_s, args.normalize_states
                            ),
                            env.normalize_obs(
                                learn_batch.first.individual_obs, args.normalize_obs
                            ),
                            method="softmax",
                        )
                        prev_vec_prob_a = jnp.sum(
                            prob_a
                            * jax.nn.one_hot(
                                learn_batch.first.action, prob_a.shape[-1]
                            ),
                            axis=-1,
                        )
                        r_term += (
                            args.alpha
                            * args.tau
                            * jnp.log(jnp.clip(prev_vec_prob_a, MIN_ACTION_PROB))
                        )

                        _, next_prob_a = q_net.apply(
                            ts.prev_q_net_params,
                            env.normalize_individual_s(
                                learn_batch.second.individual_s, args.normalize_states
                            ),
                            env.normalize_obs(
                                learn_batch.second.individual_obs, args.normalize_obs
                            ),
                            method="softmax",
                        )
                        q_next_target_term = jnp.sum(
                            next_prob_a
                            * (
                                q_next_target_vals
                                - args.tau
                                * jnp.log(jnp.clip(next_prob_a, MIN_ACTION_PROB))
                            ),
                            axis=-1,
                        )
                    else:
                        q_next_target_term = jnp.max(q_next_target_vals, axis=-1)

                    target = (
                        r_term
                        + (1 - learn_batch.first.done)
                        * args.discount_factor
                        * q_next_target_term
                    )
                    if args.loss == "mse":
                        return jnp.mean((chosen_action_q_vals - target) ** 2)
                    elif args.loss == "huber":
                        return jnp.mean(
                            rlax.huber_loss(
                                chosen_action_q_vals - target, args.huber_loss_parameter
                            )
                        )
                    else:
                        raise ValueError(f"Invalid loss: {args.loss}")

                # --- calculate loss and update network ---
                loss, grads = jax.value_and_grad(_loss_fn)(ts.params)
                ts = ts.apply_gradients(grads=grads)
                ts = ts.replace(n_updates=ts.n_updates + 1)

                # --- log loss ---
                if args.debug:

                    def log_loss(update_step, loss):
                        wandb_log_info(
                            {
                                f"{args.task}/update_step": float(update_step),
                                f"{args.task}/loss": float(loss),
                            }
                        )
                        return

                    jax.debug.callback(log_loss, ts.n_updates, loss)
                return ts, loss

            # --- update q network ---
            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (
                    ts.steps * args.num_envs * args.num_agents_per_env
                    > args.min_buffer_to_learn
                )
                & (ts.steps % args.learn_every == 0)
            )
            ts, loss = jax.lax.cond(
                is_learn_time,
                lambda ts, rng: _update_q_net(ts, rng),
                lambda ts, rng: (ts, jnp.array(0.0)),
                ts,
                _rng,
            )

            # --- update target network ---
            ts = jax.lax.cond(
                ts.steps % args.update_target_every == 0,
                lambda ts: ts.replace(target_q_net_params=ts.params),
                lambda ts: ts,
                operand=ts,
            )
            runner_state = (
                ts,
                buffer_state,
                env_mf_sequence,
                individual_s,
                individual_obs,
                rng,
            )
            return runner_state, None

        ts, buffer_state, rng = iteration_runner_state

        # --- generate mean-field sequence ---
        rng, rng_mf_sequence, rng_reset = jax.random.split(rng, 3)
        agent_wrapper = SAQNetWrapper(
            q_net,
            ts.params,
            env.normalize_obs,
            args.normalize_obs,
            env.normalize_individual_s,
            args.normalize_states,
        )
        env_mf_sequence = mf_sequence(
            rng_mf_sequence,
            env,
            agent_wrapper,
            num_envs=args.num_envs,
            max_steps_in_episode=env.params.max_steps_in_episode + 1,
        )

        # --- update previous q network params ---
        ts = ts.replace(prev_q_net_params=ts.params)

        # --- reset replay buffer ---
        if args.reset_replay_buffer_on_update:
            buffer_state = buffer.init(_transition)

        # --- reset environment ---
        rng_reset = jax.random.split(rng_reset, args.num_envs)
        individual_obs, individual_s = jax.vmap(_reset_single_env, in_axes=(0, 1))(
            rng_reset, env_mf_sequence
        )

        # --- train iteration ---
        runner_state = (
            ts,
            buffer_state,
            env_mf_sequence,
            individual_s,
            individual_obs,
            rng,
        )
        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, None, args.num_steps_per_iteration
        )
        ts, buffer_state, _, _, _, rng = runner_state
        return (ts, buffer_state, rng), None

    return _train_step


if __name__ == "__main__":
    args = tyro.cli(args)
    print("Task: ", args.task, "Algo: ", args.algo)
    args.time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- initialise logging ---
    if args.log:
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_team,
            group=args.wandb_group,
            job_type=f"{args.task}/{args.algo}",
        )
        print("Logging to wandb")
        train_time_key = f"{args.task}/train_time"
        update_step_key = f"{args.task}/update_step"
        iteration_key = f"{args.task}/iteration"
        # --- eval metrics ---
        wandb.define_metric(train_time_key)
        wandb.define_metric(iteration_key)
        wandb.define_metric(f"{args.task}/exploitability", step_metric=train_time_key)
        wandb.define_metric(
            f"{args.task}/policy_disc_return", step_metric=train_time_key
        )
        # --- debug metrics ---
        wandb.define_metric(update_step_key)
        wandb.define_metric(f"{args.task}/loss", step_metric=update_step_key)
        print("Logging to wandb")

    env = make_env(
        "sample/" + args.task,
        common_noise=args.common_noise,
    )

    def _reset_single_env(rng, env_mf_sequence):
        rng_reset_agent = jax.random.split(rng, args.num_agents_per_env)
        individual_obs, individual_s = jax.vmap(env.sa_reset, in_axes=(0, None))(
            rng_reset_agent, env_mf_sequence
        )
        return individual_obs, individual_s

    # --- make pushforward environment for evaluation purposes only ---
    if args.evaluate:
        pushforward_env = make_env(
            "pushforward/" + args.task,
            common_noise=args.common_noise,
        )
    else:
        pushforward_env = None

    # --- save args ---
    args.num_steps_per_iteration = args.num_updates_per_iteration * args.learn_every
    assert args.replay_buffer_capacity > (
        args.num_envs * args.num_agents_per_env * env.params.max_steps_in_episode
    ), (
        "replay_buffer_capacity < num_envs * num_agents_per_env * max_steps_in_episode, buffer will not contain full trajectories"
    )
    assert args.update_target_every < args.num_steps_per_iteration, (
        "update_target_every > num_steps_per_iteration, so target network remains stale for an entire iteration"
    )
    args.total_steps = args.num_iterations * args.num_steps_per_iteration
    args.epsilon_decay_steps = int(args.total_steps * args.epsilon_decay_duration_pct)
    if args.debug and not args.log:
        args.log = True
        print("Debug mode requires logging, setting log to True")
    if args.save:
        os.makedirs(f"runs/{args.task}/{args.algo}", exist_ok=True)
        with open(f"runs/{args.task}/{args.algo}/args.json", "w") as f:
            json.dump(asdict(args), f)

    # --- make single-agent q network (mean-field q network for evaluation only) ---
    q_net_kwargs = dict(
        n_actions=env.n_actions,
        state_type=args.state_type,
        num_states=env.n_states,
        tau=args.tau,
        alpha=args.alpha,
        activation=args.activation,
        hidden_layer_sizes=(128, 128, 128),
    )
    if args.q_net_type == "discrete":
        q_net = DiscreteQNet(**q_net_kwargs)
    else:
        raise ValueError(f"Invalid q_net_type: {args.q_net_type}. Expected 'discrete'.")
    mf_q_net = MeanFieldQNet(
        state_type=args.state_type,
        num_states=env.n_states,
        q_net_type=args.q_net_type,
        q_net_kwargs=q_net_kwargs,
    )

    # --- initialise single-agent q network ---
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_q_params = jax.random.split(rng, 2)
    init_individual_obs = jnp.ones(
        (args.num_envs, env.obs_dim), dtype=jnp.float32
    )  # [batch, features]
    if args.state_type == "indices":
        dummy_individual_s = jnp.ones(args.num_envs, dtype=jnp.int32)
    elif args.state_type == "states":
        dummy_individual_s = jnp.ones(
            (args.num_envs, env.individual_s_dim), dtype=jnp.float32
        )
    q_net_params = q_net.init(
        rng_q_params,
        env.normalize_individual_s(dummy_individual_s, args.normalize_states),
        env.normalize_obs(init_individual_obs, args.normalize_obs),
    )

    # --- initialise train state ---
    if args.anneal_lr:
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(
                learning_rate=optax.linear_schedule(
                    args.lr, args.lr * 0.1, args.num_iterations
                ),
                eps=1e-8,
            ),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.lr, eps=1e-8),
        )
    ts = TrainState.create(
        apply_fn=q_net.apply,
        params=q_net_params,
        prev_q_net_params=jax.tree.map(lambda x: jnp.copy(x), q_net_params),
        target_q_net_params=jax.tree.map(lambda x: jnp.copy(x), q_net_params),
        tx=tx,
        steps=0,
        n_updates=0,
    )

    # --- initialise replay buffer ---
    rng, rng_buffer, rng_buffer_dummy, _rng_buffer_dummy, _rng_buffer_dummy_ = (
        jax.random.split(rng, 5)
    )
    buffer = fbx.make_flat_buffer(
        max_length=args.replay_buffer_capacity,
        min_length=args.batch_size,
        sample_batch_size=args.batch_size,
        add_sequences=False,
        add_batch_size=args.num_envs * args.num_agents_per_env,
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )
    dummy_agent_wrapper = SAQNetWrapper(
        q_net,
        ts.params,
        env.normalize_obs,
        args.normalize_obs,
        env.normalize_individual_s,
        args.normalize_states,
    )
    dummy_env_mf_sequence = mf_sequence(
        rng_buffer_dummy,
        env,
        dummy_agent_wrapper,
        num_envs=args.num_envs,
        max_steps_in_episode=env.params.max_steps_in_episode + 1,
    )
    _individual_obs, _individual_s = env.sa_reset(
        _rng_buffer_dummy, jax.tree.map(lambda x: x[:, 1], dummy_env_mf_sequence)
    )
    _action_idx = jnp.array(0).astype(jnp.int32)
    (
        _individual_obs,
        _individual_s,
        _reward,
        _aggregate_terminated,
        _aggregate_truncated,
    ) = env.sa_step(
        _rng_buffer_dummy_,
        jax.tree.map(lambda x: x[:, 1], dummy_env_mf_sequence),
        _individual_s,
        _action_idx,
    )
    _transition = Transition(
        individual_obs=_individual_obs,
        individual_s=_individual_s.state,
        action=_action_idx,
        reward=_reward,
        done=jnp.logical_or(_aggregate_terminated, _aggregate_truncated),
    )
    buffer_state = buffer.init(_transition)

    # --- train train iteration step ---
    train_step = jax.jit(make_train_step(args, env, q_net))

    # --- initialise agent and jitted exploitability function ---
    if args.save or args.evaluate:
        mf_agent_wrapper = MFQNetWrapper(
            mf_q_net,
            ts.params,
            env.normalize_obs,
            args.normalize_obs,
            env.normalize_individual_s,
            args.normalize_states,
        )
        exploitability = make_exploitability(
            env=pushforward_env,
            agent=mf_agent_wrapper,
            state_type=args.state_type,
            gamma=args.discount_factor,
            num_envs=args.num_envs,
            max_steps_in_episode=env.params.max_steps_in_episode,
        )

    # --- train loop ---
    elapsed_time = 0.0
    iteration_idx = 0

    # --- evaluate policy ---
    if args.evaluate:
        rng, eval_rng = jax.random.split(rng)
        mf_eval_results = exploitability(eval_rng, ts.params)
        if args.log:
            wandb.log(
                {
                    f"{args.task}/iteration": float(iteration_idx),
                    f"{args.task}/update_step": float(ts.n_updates),
                    f"{args.task}/train_time": float(elapsed_time),
                    f"{args.task}/exploitability": float(
                        mf_eval_results.exploitability.exploitability
                    ),
                    f"{args.task}/policy_disc_return": float(
                        mf_eval_results.exploitability.mean_policy_return
                    ),
                }
            )
        jax.debug.print(
            "Iteration: {}, Train Time: {}, Exploitability: {}",
            iteration_idx,
            elapsed_time,
            mf_eval_results.exploitability.exploitability,
        )

    for iteration_idx in range(0, args.num_iterations, args.eval_frequency):
        iteration_idx += args.eval_frequency
        time_start = time.perf_counter()
        iteration_runner_state = (ts, buffer_state, rng)
        iteration_runner_state, _ = jax.lax.scan(
            train_step, iteration_runner_state, None, args.eval_frequency
        )
        (ts, buffer_state, rng) = iteration_runner_state
        jax.block_until_ready(ts.step)
        time_end = time.perf_counter()
        elapsed_time += time_end - time_start

        # --- evaluate policy ---
        if args.evaluate:
            rng, eval_rng = jax.random.split(rng)
            mf_eval_results = exploitability(eval_rng, ts.params)
            if args.log:
                wandb.log(
                    {
                        f"{args.task}/iteration": float(iteration_idx),
                        f"{args.task}/update_step": float(ts.n_updates),
                        f"{args.task}/train_time": float(elapsed_time),
                        f"{args.task}/exploitability": float(
                            mf_eval_results.exploitability.exploitability
                        ),
                        f"{args.task}/policy_disc_return": float(
                            mf_eval_results.exploitability.mean_policy_return
                        ),
                    }
                )
            jax.debug.print(
                "Iteration: {}, Train Time: {}, Exploitability: {}",
                iteration_idx,
                elapsed_time,
                mf_eval_results.exploitability.exploitability,
            )

    # --- save single-agent q network wrapper ---
    if args.save:
        agent_wrapper = SAQNetWrapper(
            q_net,
            ts.params,
            env.normalize_obs,
            args.normalize_obs,
            env.normalize_individual_s,
            args.normalize_states,
        )
        save_pkl(agent_wrapper, f"runs/{args.task}/{args.algo}", "sa_agent_wrapper.pkl")
        print("Agent wrapper saved")

        # --- save mean-field q network wrapper ---
        mf_agent_wrapper = MFQNetWrapper(
            mf_q_net,
            ts.params,
            env.normalize_obs,
            args.normalize_obs,
            env.normalize_individual_s,
            args.normalize_states,
        )
        save_pkl(
            mf_agent_wrapper, f"runs/{args.task}/{args.algo}", "mf_agent_wrapper.pkl"
        )
        print("MF agent wrapper saved")

    # --- finish logging ---
    if args.log:
        wandb.finish()
