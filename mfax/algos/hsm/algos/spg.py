from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
import time
from typing import NamedTuple
import tyro

from flax.training.train_state import TrainState
import gymnax
import jax
import jax.numpy as jnp
import optax
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

from mfax.algos.hsm.exploitability import make_exploitability  # noqa: E402
from mfax.algos.hsm.utils.make_act import MFActorWrapper  # noqa: E402
from mfax.algos.hsm.utils.mf_policy_wrappers import (
    MeanFieldPolicy,
    MeanFieldContinuousPolicy,
)  # noqa: E402
from mfax.envs import make_env  # noqa: E402
from mfax.envs.pushforward.base import PushforwardAggregateState  # noqa: E402
from utils import wandb_log_info, save_pkl  # noqa: E402


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
    task: str = "beach_bar_1d"
    state_type: str = "indices"
    discount_factor: float = 0.99
    normalize_obs: bool = True
    normalize_states: bool = True
    common_noise: bool = True

    # --- hsm hyperparameters ---
    algo: str = "hsm_spg"
    seed: int = 0
    num_envs: int = 128
    num_iterations: int = 20000
    lr: float = 0.001
    anneal_lr: bool = True
    max_grad_norm: float = 1.0
    activation: str = "relu"

    # --- logging ---
    debug_frequency: int = 10
    eval_frequency: int = 2000


class Transition(NamedTuple):
    aggregate_obs: jnp.ndarray  # (prices)
    aggregate_s: PushforwardAggregateState
    mat_r: jnp.ndarray
    prob_a: jnp.ndarray


def make_train_step(args, env, mf_policy_net, individual_states):

    def _train_step(runner_state, unused):

        def _mf_rollout(actor_params, rng):
            # --- start each rollout from reset ---
            rng, reset_rng, loop_rng = jax.random.split(rng, 3)
            reset_rng = jax.random.split(reset_rng, args.num_envs)
            aggregate_obs, aggregate_s = jax.vmap(env.mf_reset, in_axes=(0))(reset_rng)
            runner_state = (aggregate_s, aggregate_obs, loop_rng)

            # --- collect trajectories ---
            def _mf_transition(runner_state, unused):
                aggregate_s, aggregate_obs, rng = runner_state
                rng, rng_env, rng_next = jax.random.split(rng, 3)

                # --- select action ---
                prob_a = mf_policy_net.dist_prob(
                    actor_params,
                    env.normalize_individual_s(
                        individual_states, args.normalize_states
                    ),
                    env.normalize_obs(aggregate_obs, args.normalize_obs),
                )

                # --- step environment ---
                rng_env = jax.random.split(rng_env, args.num_envs)
                (
                    next_aggregate_obs,
                    _,
                    next_aggregate_s,
                    _,
                    mat_r,
                    terminated,
                    truncated,
                    _,
                ) = jax.vmap(env.mf_step, in_axes=(0, 0, 0))(
                    rng_env, aggregate_s, prob_a
                )
                transition = Transition(aggregate_obs, aggregate_s, mat_r, prob_a)

                runner_state = (next_aggregate_s, next_aggregate_obs, rng_next)
                return runner_state, transition

            _, transitions = jax.lax.scan(
                _mf_transition, runner_state, None, env.params.max_steps_in_episode
            )
            return transitions

        def _expected_return(traj_batch):
            def _get_expected_return(disc_rewards, transition):
                prob_a, mat_r, aggregate_s = (
                    transition.prob_a,
                    transition.mat_r,
                    transition.aggregate_s,
                )
                expected = jax.vmap(env.mf_expected_value, in_axes=(0, 0, 0))

                disc_rewards = jnp.sum(
                    prob_a * mat_r, axis=-1
                ) + args.discount_factor * expected(disc_rewards, prob_a, aggregate_s)
                return disc_rewards, None

            disc_rewards, _ = jax.lax.scan(
                _get_expected_return,
                jnp.zeros_like(traj_batch.aggregate_s.mu[0]),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            mf_weighted_disc_rewards = jnp.sum(
                disc_rewards * traj_batch.aggregate_s.mu[0], axis=-1
            )
            return mf_weighted_disc_rewards

        # --- loss is just negative expected return ---
        def _loss(actor_params, rng):
            transitions = _mf_rollout(actor_params, rng)
            return -_expected_return(transitions).mean()

        # --- for debugging ---
        def _grad_norm_stats(grads):
            per_leaf_norms = jax.tree.map(lambda g: jnp.linalg.norm(g), grads)
            norms = jnp.stack(jax.tree_util.tree_leaves(per_leaf_norms))
            return norms.mean(), norms.max()

        # --- update networks ---
        actor_ts, rng = runner_state
        rng, _rng = jax.random.split(rng)
        actor_loss, actor_grad = jax.value_and_grad(_loss)(actor_ts.params, _rng)
        if args.debug:
            actor_grad_norm_mean, actor_grad_norm_max = _grad_norm_stats(actor_grad)

            def log_losses(
                update_step, actor_loss, actor_grad_norm_mean, actor_grad_norm_max
            ):
                wandb_log_info(
                    {
                        f"{args.task}/update_step": float(update_step),
                        f"{args.task}/actor_loss": float(actor_loss),
                        f"{args.task}/actor_grad_norm_mean": float(
                            actor_grad_norm_mean
                        ),
                        f"{args.task}/actor_grad_norm_max": float(actor_grad_norm_max),
                    }
                )
                return

            jax.lax.cond(
                actor_ts.step % args.debug_frequency == 0,
                lambda: jax.debug.callback(
                    log_losses,
                    actor_ts.step,
                    actor_loss,
                    actor_grad_norm_mean,
                    actor_grad_norm_max,
                ),
                lambda: None,
            )
        actor_ts = actor_ts.apply_gradients(grads=actor_grad)
        runner_state = (actor_ts, rng)
        return runner_state, None

    return _train_step


if __name__ == "__main__":
    args = tyro.cli(args)
    args.time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.state_type == "indices" and args.normalize_states:
        args.normalize_states = False
        print(
            "Normalization of individual states is not supported for indices, setting normalize_states to False"
        )
    if args.debug and not args.log:
        args.log = True
        print("Debug mode requires logging, setting log to True")
    if args.save:
        os.makedirs(f"runs/{args.task}/{args.algo}", exist_ok=True)
        with open(f"runs/{args.task}/{args.algo}/args.json", "w") as f:
            json.dump(asdict(args), f)

    # --- initialise logging ---
    if args.log:
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_team,
            group=args.wandb_group,
            job_type=f"{args.task}/{args.algo}",
        )
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
        wandb.define_metric(f"{args.task}/actor_loss", step_metric=update_step_key)
        wandb.define_metric(
            f"{args.task}/actor_grad_norm_mean", step_metric=update_step_key
        )
        wandb.define_metric(
            f"{args.task}/actor_grad_norm_max", step_metric=update_step_key
        )
        print("Logging to wandb")

    # --- make environment ---
    env = make_env(
        "pushforward/" + args.task,
        common_noise=args.common_noise,
    )

    # --- make mean-field policy network ---
    if isinstance(env.action_space, gymnax.environments.spaces.Discrete):
        policy_kwargs = dict(
            n_actions=env.n_actions,
            hidden_layer_sizes=(128, 128, 128),
            activation=args.activation,
        )
        mf_policy_net = MeanFieldPolicy(
            state_type=args.state_type,
            num_states=env.n_states,
            policy_kwargs=policy_kwargs,
        )
    else:
        assert isinstance(env.action_space, gymnax.environments.spaces.Box), (
            f"Invalid action space: {env.action_space}"
        )
        policy_kwargs = dict(
            action_dim=env.action_space.shape[-1],
            action_range=(env.action_space.low, env.action_space.high),
            hidden_layer_sizes=(128, 128, 128),
            activation=args.activation,
            state_type=args.state_type,
            num_states=env.n_states,
        )
        mf_policy_net = MeanFieldContinuousPolicy(
            state_type=args.state_type,
            num_states=env.n_states,
            actions=env.params.discrete_actions,
            policy_kwargs=policy_kwargs,
        )

    # --- individual states ---
    if args.state_type == "indices":
        individual_states = jnp.arange(env.n_states)
    elif args.state_type == "states":
        individual_states = env.params.states
    else:
        raise ValueError(f"Invalid state type: {args.state_type}")

    # --- initialise actor ---
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_actor_params, rng_action = jax.random.split(rng, 3)
    init_obs = jnp.ones(
        (args.num_envs, env.obs_dim), dtype=jnp.float32
    )  # [batch, features]
    actor_params = mf_policy_net.init(
        rng_actor_params,
        env.normalize_individual_s(individual_states, args.normalize_states),
        env.normalize_obs(init_obs, args.normalize_obs),
        rng_action,
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
    actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)

    # --- make train step ---
    train_step = jax.jit(make_train_step(args, env, mf_policy_net, individual_states))

    # --- initialise agent and jitted exploitability function ---
    if args.save or args.evaluate:
        mf_agent_wrapper = MFActorWrapper(
            mf_policy_net,
            actor_ts.params,
            env.normalize_obs,
            args.normalize_obs,
            env.normalize_individual_s,
            args.normalize_states,
        )
        exploitability = make_exploitability(
            env=env,
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
        mf_eval_results = exploitability(eval_rng, actor_ts.params)
        if args.log:
            wandb.log(
                {
                    f"{args.task}/iteration": float(iteration_idx),
                    f"{args.task}/update_step": float(actor_ts.step),
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
        iteration_runner_state = (actor_ts, rng)
        iteration_runner_state, _ = jax.lax.scan(
            train_step, iteration_runner_state, None, args.eval_frequency
        )
        (actor_ts, rng) = iteration_runner_state
        jax.block_until_ready(iteration_runner_state)
        time_end = time.perf_counter()
        elapsed_time += time_end - time_start

        # --- evaluate policy ---
        if args.evaluate:
            rng, eval_rng = jax.random.split(rng)
            mf_eval_results = exploitability(eval_rng, actor_ts.params)
            if args.log:
                wandb.log(
                    {
                        f"{args.task}/iteration": float(iteration_idx),
                        f"{args.task}/update_step": float(actor_ts.step),
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

    # --- save mf agent wrapper ---
    if args.save:
        mf_agent_wrapper = MFActorWrapper(
            mf_policy_net,
            actor_ts.params,
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
