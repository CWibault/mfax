from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
import time
import tyro
import math

import chex
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

from mfax.algos.rl.sequence import mf_sequence  # noqa: E402
from mfax.algos.rl.utils.make_act import (
    SARecurrentActorWrapper,
    SARecurrentValueWrapper,
)  # noqa: E402
from mfax.envs import make_env  # noqa: E402
from mfax.algos.rl.utils.sa_policy_wrappers import RecurrentSingleAgentPolicy  # noqa: E402
from mfax.algos.rl.utils.sa_value_wrappers import RecurrentSingleAgentValue  # noqa: E402
from utils import wandb_log_info, save_pkl  # noqa: E402

# --- for evaluation only, training uses sampled-based mean-field ---
from mfax.algos.hsm.utils.mf_policy_wrappers import (
    RecurrentMeanFieldPolicy,
    RecurrentMeanFieldContinuousPolicy,
)  # noqa: E402
from mfax.algos.hsm.utils.mf_value_wrappers import RecurrentMeanFieldValue  # noqa: E402
from mfax.algos.hsm.utils.make_act import (
    MFRecurrentActorWrapper,
    MFRecurrentValueWrapper,
)  # noqa: E402
from mfax.algos.hsm.exploitability import make_exploitability  # noqa: E402


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

    # --- PPO hyperparameters ---
    algo: str = "rl_rippo"
    seed: int = 0
    num_envs: int = 128
    num_agents_per_env: int = 64
    num_steps: int = 64
    num_epochs: int = 1
    num_minibatches: int = 8
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 0.001
    anneal_lr: bool = True
    max_grad_norm: float = 1.0
    activation: str = "relu"

    # --- iterations ---
    num_iterations: int = 50
    # --- number of gradient update steps per iteration ---
    num_updates_per_iteration: int = 200
    eval_frequency: int = 10


@chex.dataclass(frozen=True)
class Transition:
    individual_obs: chex.Array
    individual_s: chex.Array
    done: chex.Array
    action: chex.Array
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: chex.Array


def make_train_step(args, env, policy_net, value_net):

    def _train_step(iteration_runner_state, unused):

        def _train_iteration_step(runner_state, unused):

            # --- collect trajectories ---
            def _env_step(runner_state, unused):
                (
                    actor_ts,
                    critic_ts,
                    env_mf_sequence,
                    last_individual_s,
                    last_individual_obs,
                    last_done,
                    last_actor_hidden,
                    last_critic_hidden,
                    rng,
                ) = runner_state

                def _agent_step(
                    env_mf_sequence,
                    last_individual_s,
                    last_individual_obs,
                    last_done,
                    last_actor_hidden,
                    last_critic_hidden,
                    rng,
                ):
                    # --- select action ---
                    rng, _rng = jax.random.split(rng)
                    (action, log_prob), actor_hidden = policy_net.sample_and_log_prob(
                        actor_ts.params,
                        env.normalize_individual_s(
                            last_individual_s.state, args.normalize_states
                        ),
                        env.normalize_obs(last_individual_obs, args.normalize_obs),
                        last_actor_hidden,
                        last_done,
                        _rng,
                    )
                    value, critic_hidden = value_net(
                        critic_ts.params,
                        env.normalize_individual_s(
                            last_individual_s.state, args.normalize_states
                        ),
                        env.normalize_obs(last_individual_obs, args.normalize_obs),
                        last_critic_hidden,
                        last_done,
                    )

                    # --- step environment ---
                    rng_step_agent = jax.random.split(_rng, args.num_agents_per_env)
                    (
                        individual_obs,
                        individual_s,
                        reward,
                        aggregate_terminated,
                        aggregate_truncated,
                    ) = jax.vmap(env.sa_step, in_axes=(0, None, 0, 0))(
                        rng_step_agent, env_mf_sequence, last_individual_s, action
                    )
                    done = jnp.logical_or(aggregate_terminated, aggregate_truncated)
                    transition = Transition(
                        individual_s=last_individual_s.state,
                        individual_obs=last_individual_obs,
                        done=last_done,
                        action=action,
                        log_prob=log_prob,
                        value=value,
                        reward=reward,
                    )
                    return (
                        transition,
                        individual_s,
                        individual_obs,
                        done,
                        actor_hidden,
                        critic_hidden,
                    )

                rng, _rng = jax.random.split(rng)
                rng_step_env = jax.random.split(_rng, args.num_envs)
                (
                    transition,
                    individual_s,
                    individual_obs,
                    done,
                    actor_hidden,
                    critic_hidden,
                ) = jax.vmap(_agent_step, in_axes=(1, 0, 0, 0, 0, 0, 0))(
                    env_mf_sequence,
                    last_individual_s,
                    last_individual_obs,
                    last_done,
                    last_actor_hidden,
                    last_critic_hidden,
                    rng_step_env,
                )
                runner_state = (
                    actor_ts,
                    critic_ts,
                    env_mf_sequence,
                    individual_s,
                    individual_obs,
                    done,
                    actor_hidden,
                    critic_hidden,
                    rng,
                )
                return runner_state, transition

            # --- save initial hidden states for rerunning networks ---
            init_actor_hidden = runner_state[-3]
            init_critic_hidden = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, args.num_steps
            )

            # --- calculate advantage ---
            (
                actor_ts,
                critic_ts,
                env_mf_sequence,
                last_individual_s,
                last_individual_obs,
                last_done,
                last_actor_hidden,
                last_critic_hidden,
                rng,
            ) = runner_state
            last_value, _ = jax.vmap(value_net, in_axes=(None, 0, 0, 0, 0))(
                critic_ts.params,
                env.normalize_individual_s(
                    last_individual_s.state, args.normalize_states
                ),
                env.normalize_obs(last_individual_obs, args.normalize_obs),
                last_critic_hidden,
                last_done,
            )

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(gae_next_value_next_done, transition):
                    gae, next_value, next_done = gae_next_value_next_done
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward
                        + args.discount_factor * next_value * (1 - next_done)
                        - value
                    )  # TODO: Should be terminated if truncated - and then the value should be the value of the non-reset state
                    gae = (
                        delta
                        + args.discount_factor * args.gae_lambda * (1 - next_done) * gae
                    )
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_value, last_done)

            # --- update network ---
            def _update_epoch(update_state, unused):

                def _update_minibatch(train_states_rng, batch_info):
                    actor_ts, critic_ts, rng = train_states_rng
                    (
                        init_actor_hidden,
                        init_critic_hidden,
                        traj_batch,
                        advantages,
                        targets,
                    ) = batch_info
                    assert init_actor_hidden.shape == (
                        1,
                        args.num_envs * args.num_agents_per_env // args.num_minibatches,
                        policy_net.encoder.hidden_size,
                    ), f"init_actor_hidden shape: {init_actor_hidden.shape}"

                    def actor_loss_fn(actor_params, traj_batch, gae):
                        # --- rerun actor network ---
                        individual_s = env.normalize_individual_s(
                            traj_batch.individual_s, args.normalize_states
                        )
                        individual_obs = env.normalize_obs(
                            traj_batch.individual_obs, args.normalize_obs
                        )

                        def _actor_scan_step(hidden, inputs):
                            action, done, individual_s, individual_obs = inputs
                            (log_prob, entropy), next_hidden = (
                                policy_net.log_prob_entropy(
                                    actor_params,
                                    individual_s,
                                    individual_obs,
                                    hidden,
                                    done,
                                    action,
                                )
                            )
                            return next_hidden, (log_prob, entropy)

                        _, (log_prob, entropy) = jax.lax.scan(
                            _actor_scan_step,
                            init_actor_hidden[
                                0
                            ],  # removing time dimension that was added for aligning indices in permutation
                            (
                                traj_batch.action,
                                traj_batch.done,
                                individual_s,
                                individual_obs,
                            ),
                            unroll=16,
                        )
                        entropy = entropy.mean()

                        # --- calculate actor loss ---
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()

                        total_loss = loss_actor - args.ent_coef * entropy
                        return total_loss, (loss_actor, entropy)

                    def critic_loss_fn(critic_params, traj_batch, targets):
                        # --- rerun critic ---
                        individual_s = env.normalize_individual_s(
                            traj_batch.individual_s, args.normalize_states
                        )
                        individual_obs = env.normalize_obs(
                            traj_batch.individual_obs, args.normalize_obs
                        )

                        def _critic_scan_step(hidden, inputs):
                            done, individual_s, individual_obs = inputs
                            value, next_hidden = value_net(
                                critic_params,
                                individual_s,
                                individual_obs,
                                hidden,
                                done,
                            )
                            return next_hidden, value

                        _, value = jax.lax.scan(
                            _critic_scan_step,
                            init_critic_hidden[
                                0
                            ],  # removing time dimension that was added for aligning indices in permutation
                            (traj_batch.done, individual_s, individual_obs),
                            unroll=16,
                        )
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-args.clip_eps, args.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        return value_loss

                    actor_losses, actor_grads = jax.value_and_grad(
                        actor_loss_fn, has_aux=True
                    )(actor_ts.params, traj_batch, advantages)
                    critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
                        critic_ts.params, traj_batch, targets
                    )
                    actor_ts = actor_ts.apply_gradients(grads=actor_grads)
                    critic_ts = critic_ts.apply_gradients(grads=critic_grads)

                    # --- log losses ---
                    _, (actor_loss, entropy) = actor_losses
                    if args.debug:

                        def log_losses(update_step, actor_loss, critic_loss, entropy):
                            wandb_log_info(
                                {
                                    f"{args.task}/update_step": float(update_step),
                                    f"{args.task}/actor_loss": float(actor_loss),
                                    f"{args.task}/critic_loss": float(critic_loss),
                                    f"{args.task}/entropy": float(entropy),
                                }
                            )
                            return

                        jax.debug.callback(
                            log_losses, actor_ts.step, actor_loss, critic_loss, entropy
                        )

                    return (actor_ts, critic_ts, rng), (
                        actor_loss,
                        critic_loss,
                        entropy,
                    )

                (
                    actor_ts,
                    critic_ts,
                    init_actor_hidden,
                    init_critic_hidden,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                # --- shuffle batches across environments ---
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(
                    _rng, args.num_envs * args.num_agents_per_env
                )
                batch = (
                    init_actor_hidden[None, :],
                    init_critic_hidden[None, :],
                    traj_batch,
                    advantages,
                    targets,
                )  # add time dimension to the initial hidden states for permutation (init_actor_hidden goes from (num_envs, num_agents_per_env, hidden_size), to (1, num_envs, num_agents_per_env, hidden_size)).
                flattened_batch = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x,
                        tuple(
                            [x.shape[0], args.num_envs * args.num_agents_per_env]
                            + list(x.shape[3:])
                        ),
                    ),
                    batch,
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), flattened_batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            tuple(
                                [x.shape[0], args.num_minibatches, -1]
                                + list(x.shape[2:])
                            ),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                # --- update network with minibatches ---
                (actor_ts, critic_ts, rng), losses = jax.lax.scan(
                    _update_minibatch, (actor_ts, critic_ts, rng), minibatches
                )
                update_state = (
                    actor_ts,
                    critic_ts,
                    init_actor_hidden,
                    init_critic_hidden,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            update_state = (
                actor_ts,
                critic_ts,
                init_actor_hidden,
                init_critic_hidden,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, _ = jax.lax.scan(
                _update_epoch, update_state, None, args.num_epochs
            )
            (actor_ts, critic_ts, _, _, _, _, _, rng) = update_state
            # --- reinitialise hidden states after having updated networks ---
            runner_state = (
                actor_ts,
                critic_ts,
                env_mf_sequence,
                last_individual_s,
                last_individual_obs,
                last_done,
                init_actor_hidden,
                init_critic_hidden,
                rng,
            )
            return runner_state, None

        actor_ts, critic_ts, rng = iteration_runner_state

        # --- generate mean-field sequence ---
        rng, rng_mf_sequence, rng_reset = jax.random.split(rng, 3)
        agent_wrapper = SARecurrentActorWrapper(
            policy_net,
            actor_ts.params,
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

        # --- reset environment ---
        rng_reset = jax.random.split(rng_reset, args.num_envs)
        individual_obs, individual_s = jax.vmap(_reset_single_env, in_axes=(0, 1))(
            rng_reset, env_mf_sequence
        )

        # --- initialise hidden states and done flags ---
        init_actor_hidden = policy_net.init_hidden(
            args.num_envs * args.num_agents_per_env, policy_net.encoder.hidden_size
        ).reshape((args.num_envs, args.num_agents_per_env, -1))
        init_critic_hidden = value_net.init_hidden(
            args.num_envs * args.num_agents_per_env, value_net.encoder.hidden_size
        ).reshape((args.num_envs, args.num_agents_per_env, -1))
        init_done = jnp.zeros((args.num_envs, args.num_agents_per_env), dtype=jnp.bool_)

        # --- train iteration ---
        runner_state = (
            actor_ts,
            critic_ts,
            env_mf_sequence,
            individual_s,
            individual_obs,
            init_done,
            init_actor_hidden,
            init_critic_hidden,
            rng,
        )
        runner_state, _ = jax.lax.scan(
            _train_iteration_step,
            runner_state,
            None,
            math.ceil(
                args.num_updates_per_iteration
                / (args.num_minibatches * args.num_epochs)
            ),
        )
        (actor_ts, critic_ts, _, _, _, _, _, _, rng) = runner_state
        return (actor_ts, critic_ts, rng), None

    return _train_step


if __name__ == "__main__":
    args = tyro.cli(args)
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
        wandb.define_metric(f"{args.task}/critic_loss", step_metric=update_step_key)
        wandb.define_metric(f"{args.task}/entropy", step_metric=update_step_key)
        print("Logging to wandb")

    # --- initialise environment ---
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

    # -- save args ---
    if args.debug and not args.log:
        args.log = True
        print("Debug mode requires logging, setting log to True")
    assert args.num_updates_per_iteration > (args.num_minibatches * args.num_epochs), (
        "num_updates_per_iteration must be larger than num_minibatches * num_epochs"
    )
    if args.save:
        os.makedirs(f"runs/{args.task}/{args.algo}", exist_ok=True)
        with open(f"runs/{args.task}/{args.algo}/args.json", "w") as f:
            json.dump(asdict(args), f)

    # --- make single-agent policy network (mean-field policy network for evaluation only) ---
    encoder_kwargs = dict(
        hidden_size=(128, 128, 128)[0],
        embed_size=(128, 128, 128)[0],
        activation=args.activation,
    )
    if isinstance(env.action_space, gymnax.environments.spaces.Discrete):
        policy_kwargs = dict(
            n_actions=env.n_actions,
            hidden_layer_sizes=(128, 128, 128),
            activation=args.activation,
            state_type=args.state_type,
            num_states=env.n_states,
        )
        policy_net = RecurrentSingleAgentPolicy(
            state_type=args.state_type,
            num_states=env.n_states,
            policy_type="discrete",
            policy_kwargs=policy_kwargs,
            encoder_kwargs=encoder_kwargs,
        )
        mf_policy_net = RecurrentMeanFieldPolicy(
            state_type=args.state_type,
            num_states=env.n_states,
            encoder_kwargs=encoder_kwargs,
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
        policy_net = RecurrentSingleAgentPolicy(
            state_type=args.state_type,
            num_states=env.n_states,
            policy_type="beta",
            policy_kwargs=policy_kwargs,
            encoder_kwargs=encoder_kwargs,
        )
        mf_policy_net = RecurrentMeanFieldContinuousPolicy(
            state_type=args.state_type,
            num_states=env.n_states,
            actions=env.params.discrete_actions,
            encoder_kwargs=encoder_kwargs,
            policy_kwargs=policy_kwargs,
        )

    # --- make single-agent value network (mean-field value network for evaluation only) ---
    value_kwargs = dict(
        hidden_layer_sizes=(128, 128, 128),
        activation=args.activation,
        state_type=args.state_type,
        num_states=env.n_states,
    )
    value_net = RecurrentSingleAgentValue(
        state_type=args.state_type,
        num_states=env.n_states,
        encoder_kwargs=encoder_kwargs,
        value_kwargs=value_kwargs,
    )
    mf_value_net = RecurrentMeanFieldValue(
        state_type=args.state_type,
        num_states=env.n_states,
        encoder_kwargs=encoder_kwargs,
        value_kwargs=value_kwargs,
    )

    # --- initialise single-agent policy network ---
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_policy_params, rng_value_params, rng_action = jax.random.split(rng, 4)
    dummy_individual_obs = jnp.ones((args.num_envs, env.obs_dim), dtype=jnp.float32)
    if args.state_type == "indices":
        dummy_individual_s = jnp.ones(args.num_envs, dtype=jnp.int32)
    elif args.state_type == "states":
        dummy_individual_s = jnp.ones(
            (args.num_envs, env.individual_s_dim), dtype=jnp.float32
        )
    dummy_done = jnp.zeros((args.num_envs), dtype=jnp.bool_)
    actor_params = policy_net.init(
        rng_policy_params,
        env.normalize_individual_s(dummy_individual_s, args.normalize_states),
        env.normalize_obs(dummy_individual_obs, args.normalize_obs),
        None,
        dummy_done,
        rng_action,
    )
    # --- initialise single-agent value network ---
    critic_params = value_net.init(
        rng_value_params,
        env.normalize_individual_s(dummy_individual_s, args.normalize_states),
        env.normalize_obs(dummy_individual_obs, args.normalize_obs),
        None,
        dummy_done,
    )

    # --- initialise train states ---
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
    critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)

    # --- make train-iteration step ---
    train_step = jax.jit(make_train_step(args, env, policy_net, value_net))

    # --- initialise agent and jitted exploitability function ---
    if args.save or args.evaluate:
        mf_agent_wrapper = MFRecurrentActorWrapper(
            mf_policy_net,
            actor_ts.params,
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
        iteration_runner_state = (actor_ts, critic_ts, rng)
        iteration_runner_state, _ = jax.lax.scan(
            train_step, iteration_runner_state, None, args.eval_frequency
        )
        (actor_ts, critic_ts, rng) = iteration_runner_state
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

    # --- save single agent and value wrappers ---
    if args.save:
        agent_wrapper = SARecurrentActorWrapper(
            policy_net,
            actor_ts.params,
            env.normalize_obs,
            args.normalize_obs,
            env.normalize_individual_s,
            args.normalize_states,
        )
        value_wrapper = SARecurrentValueWrapper(
            value_net,
            critic_ts.params,
            env.normalize_obs,
            args.normalize_obs,
            env.normalize_individual_s,
            args.normalize_states,
        )
        save_pkl(agent_wrapper, f"runs/{args.task}/{args.algo}", "sa_agent_wrapper.pkl")
        save_pkl(value_wrapper, f"runs/{args.task}/{args.algo}", "sa_value_wrapper.pkl")
        print("Agent wrapper saved")

        # --- save mf agent and value wrappers ---
        mf_agent_wrapper = MFRecurrentActorWrapper(
            mf_policy_net,
            actor_ts.params,
            env.normalize_obs,
            args.normalize_obs,
            env.normalize_individual_s,
            args.normalize_states,
        )
        mf_value_wrapper = MFRecurrentValueWrapper(
            mf_value_net,
            critic_ts.params,
            env.normalize_obs,
            args.normalize_obs,
            env.normalize_individual_s,
            args.normalize_states,
        )
        save_pkl(
            mf_agent_wrapper, f"runs/{args.task}/{args.algo}", "mf_agent_wrapper.pkl"
        )
        save_pkl(
            mf_value_wrapper, f"runs/{args.task}/{args.algo}", "mf_value_wrapper.pkl"
        )
        print("MF agent wrappers saved")

    # --- finish logging ---
    if args.log:
        wandb.finish()
