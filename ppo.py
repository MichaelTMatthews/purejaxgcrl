import argparse
import os
import sys
import time

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax

import wandb

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from envs.envs import create_goal_set_fns, create_env
from logz.logger import log
from models.actor_critic_gc import ActorCriticConvSymbolicCraftaxGC, ActorCriticGC
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)


@chex.dataclass(frozen=True)
class Transition:
    done: jnp.ndarray
    done_goal: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_gc: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

    goal_index: jnp.ndarray
    num_goals_completed: jnp.ndarray


def make_train(config):
    env, env_params, static_env_params = create_env(config)

    env = LogWrapper(env)

    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = BatchEnvWrapper(AutoResetEnvWrapper(env), num_envs=config["NUM_ENVS"])

    (
        goal_achieved,
        goal_indexes_to_goals,
        goal_to_goal_index,
        sample_positive_goal_index_from_obs,
        get_goals_seen,
        sample_goal,
        get_all_goals,
    ) = create_goal_set_fns(config["ENV_NAME"])

    all_goal_names, all_goals = get_all_goals(config["ENV_NAME"], static_env_params)
    num_goals = jax.tree.leaves(all_goals)[0].shape[0]
    print("num_goals", num_goals)

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_MINIBATCHES"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["MINIBATCH_SIZE"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["NUM_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # INIT NETWORK
    if config["NETWORK_TYPE"] == "symbolic_flat":
        network = ActorCriticGC(
            action_dim=env.action_space(env_params).n,
            layer_width=config["NETWORK_LAYER_WIDTH"],
            embedding_layers=config["NETWORK_NUM_EMBEDDING_LAYERS"],
            actor_layers=config["NETWORK_NUM_ACTOR_LAYERS"],
            critic_layers=config["NETWORK_NUM_CRITIC_LAYERS"],
            sigmoid_critic=config["NETWORK_SIGMOID_VALUE"],
        )
        print("Using symbolic flat network")

    elif config["NETWORK_TYPE"] == "symbolic_conv" and "Craftax" in config["ENV_NAME"]:
        network = ActorCriticConvSymbolicCraftaxGC(
            action_dim=env.action_space(env_params).n,
            layer_width=config["NETWORK_LAYER_WIDTH"],
            conv_layers=config["NETWORK_CONV_LAYERS"],
            conv_features=config["NETWORK_CONV_FEATURES"],
            conv_kernel_size=config["NETWORK_CONV_KERNEL_SIZE"],
            embedding_layers=config["NETWORK_NUM_EMBEDDING_LAYERS"],
            actor_layers=config["NETWORK_NUM_ACTOR_LAYERS"],
            critic_layers=config["NETWORK_NUM_CRITIC_LAYERS"],
            sigmoid_critic=config["NETWORK_SIGMOID_VALUE"],
        )
        print("Using symbolic conv network")
    else:
        raise ValueError(f"Unknown symbolic network type: {config['NETWORK_TYPE']}")

    def train(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        init_x = jax.tree.map(lambda x: x[:1], obsv)
        example_goal = jax.tree.map(lambda x: x[0], all_goals)
        rng, _rng = jax.random.split(rng)
        network_params = network.init(
            _rng, init_x, jax.tree.map(lambda x: x[None], example_goal)
        )

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    rng,
                    update_step,
                    goal_indexes,
                    success_counter,
                    failure_counter,
                    goals_seen,
                    num_goals_completed,
                ) = runner_state

                # Select Action
                goal_reprs = goal_indexes_to_goals(all_goals, goal_indexes)
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs, goal_reprs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # Step env
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                # GC reward
                goals_achieved = jax.vmap(goal_achieved)(obsv, goal_reprs)
                reward_gc = goals_achieved * config["GOAL_REWARD_SCALE"]

                # Add goal finishes to info
                info["goals_achieved"] = goals_achieved

                # Construct batched transition
                transition = Transition(
                    done=done,
                    done_goal=goals_achieved,
                    action=action,
                    value=value,
                    reward_e=reward_e,
                    reward_gc=reward_gc,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                    goal_index=goal_indexes,
                    num_goals_completed=num_goals_completed,
                )

                # Sample new goals for completed goals and terminated episodes
                rng, _rng = jax.random.split(rng)

                _rngs = jax.random.split(_rng, config["NUM_ENVS"])
                new_goal_indexes = jax.vmap(sample_goal, in_axes=(0, None, None))(
                    _rngs, config["ONLY_SAMPLE_FROM_SEEN_GOALS"], goals_seen
                )

                num_goals_completed = jax.tree.map(
                    lambda x, y: jax.vmap(jax.lax.select)(goals_achieved, x, y),
                    num_goals_completed + 1,
                    num_goals_completed,
                )

                num_goals_completed = jax.tree.map(
                    lambda x, y: jax.vmap(jax.lax.select)(done, x, y),
                    jnp.zeros_like(num_goals_completed),
                    num_goals_completed,
                )

                done_ep_or_goal = jnp.logical_or(done, goals_achieved)
                goal_indexes = jax.tree.map(
                    lambda x, y: jax.vmap(jax.lax.select)(done_ep_or_goal, x, y),
                    new_goal_indexes,
                    goal_indexes,
                )

                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    rng,
                    update_step,
                    goal_indexes,
                    success_counter,
                    failure_counter,
                    goals_seen,
                    num_goals_completed,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            (
                train_state,
                env_state,
                last_obs,
                rng,
                update_step,
                goal_indexes,
                success_counter,
                failure_counter,
                goals_seen,
                num_goals_completed,
            ) = runner_state

            live_success_rates = success_counter / (
                success_counter + failure_counter + 1e-7
            )

            goals_seen |= get_goals_seen(
                jax.tree.map(
                    lambda x: x.reshape((x.shape[0] * x.shape[1], *x.shape[2:])),
                    traj_batch.obs,
                ),
                all_goals,
            )

            # Update success/failure counters
            successful_goals = jax.tree.map(
                lambda y: jax.vmap(jax.vmap(jax.lax.select))(
                    jnp.logical_and(
                        traj_batch.done_goal,
                        traj_batch.num_goals_completed == 0,
                    ),
                    jax.nn.one_hot(y, num_classes=num_goals),
                    jnp.zeros_like(jax.nn.one_hot(y, num_classes=num_goals)),
                ),
                traj_batch.goal_index,
            )
            success_counter = jax.tree.map(
                lambda x, y: x + y.sum(axis=0).sum(axis=0),
                success_counter,
                successful_goals,
            )

            failed_goals = jax.tree.map(
                lambda y: jax.vmap(jax.vmap(jax.lax.select))(
                    jnp.logical_and(
                        traj_batch.done,
                        traj_batch.num_goals_completed == 0,
                    ),
                    jax.nn.one_hot(y, num_classes=num_goals),
                    jnp.zeros_like(jax.nn.one_hot(y, num_classes=num_goals)),
                ),
                traj_batch.goal_index,
            )
            failure_counter = jax.tree.map(
                lambda x, y: x + y.sum(axis=0).sum(axis=0),
                failure_counter,
                failed_goals,
            )

            success_counter = success_counter * config["LIVE_SUCCESS_RATE_DECAY"]
            failure_counter = failure_counter * config["LIVE_SUCCESS_RATE_DECAY"]

            # CALCULATE ADVANTAGES
            last_goal_repr = goal_indexes_to_goals(all_goals, goal_indexes)
            _, last_val = network.apply(train_state.params, last_obs, last_goal_repr)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done_ep, done_goal, value, reward_gc = (
                        transition.done,
                        transition.done_goal,
                        transition.value,
                        transition.reward_gc,
                    )

                    done = jax.vmap(jnp.logical_or)(done_ep, done_goal)

                    delta = (
                        reward_gc + config["GAMMA"] * next_value * (1 - done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    # Policy/value network
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        goal_repr = goal_indexes_to_goals(
                            all_goals, traj_batch.goal_index
                        )
                        pi, value = network.apply(params, traj_batch.obs, goal_repr)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, 0)
                    return train_state, losses

                (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ENVS"], (
                    "batch size must be equal to number of steps * number of envs"
                )
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["NUM_EPOCHS"]
            )

            train_state = update_state[0]

            rng = update_state[-1]
            metrics = {
                "total_loss": loss_info[0][0].mean(),
                "value_loss": loss_info[0][1][0].mean(),
                "actor_loss": loss_info[0][1][1].mean(),
                "entropy_loss": loss_info[0][1][2].mean(),
                "value_loss_scaled": loss_info[0][1][0].mean() * config["VF_COEF"],
                "entropy_loss_scaled": loss_info[0][1][2].mean() * config["ENT_COEF"],
                "update_step": update_step,
            }

            # wandb logging
            if config["USE_WANDB"]:
                for goal_index, goal_name in enumerate(all_goal_names):
                    sr = live_success_rates[goal_index]
                    metrics[f"live_success_rates_{goal_name}"] = sr

                metrics["live_success_rates_aggregate/all"] = live_success_rates.mean()

                metrics["goals/num_goals_completed"] = (
                    num_goals_completed * traj_batch.done
                ).sum() / traj_batch.done.sum()
                metrics["goals/goals_seen"] = goals_seen.sum()
                metrics["goals/goals_seen_ratio"] = goals_seen.sum() / num_goals

                def callback(metric, update_step):
                    log(metric, update_step, config)

                jax.debug.callback(
                    callback,
                    metrics,
                    update_step,
                )

            runner_state = (
                train_state,
                env_state,
                last_obs,
                rng,
                update_step + 1,
                goal_indexes,
                success_counter,
                failure_counter,
                goals_seen,
                num_goals_completed,
            )
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        init_obss, _ = env.reset(_rng, env_params)
        seen_goals = get_goals_seen(init_obss, all_goals)

        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, config["NUM_ENVS"])
        goal_indexes = jax.vmap(sample_goal, in_axes=(0, None, None))(
            _rngs, config["ONLY_SAMPLE_FROM_SEEN_GOALS"], seen_goals
        )
        num_goals_completed = jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32)

        success_counter = jnp.zeros(num_goals)
        failure_counter = jnp.zeros(num_goals)

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            _rng,
            0,
            goal_indexes,
            success_counter,
            failure_counter,
            seen_goals,
            num_goals_completed,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng = jax.random.split(rng)

    train_jit = jax.jit(make_train(config))

    t0 = time.time()
    out = jax.block_until_ready(train_jit(_rng))
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    if config["USE_WANDB"]:

        def _save_network(rs_fn, dir_name):
            train_state = rs_fn(out["runner_state"])
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(lambda x: x[0], "policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # PPO
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--minibatch_size", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--no_anneal_lr", dest="anneal_lr", action="store_false")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no_use_wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--no_use_optimistic_resets", dest="use_optimistic_resets", action="store_false"
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # Network
    parser.add_argument(
        "--network_type",
        type=str,
        default="symbolic_conv",
        choices=["symbolic_conv", "symbolic_flat"],
    )
    parser.add_argument("--network_layer_width", type=int, default=1024)
    parser.add_argument("--network_conv_layers", type=int, default=1)
    parser.add_argument("--network_conv_features", type=int, default=32)
    parser.add_argument("--network_conv_kernel_size", type=int, default=3)
    parser.add_argument("--network_num_embedding_layers", type=int, default=1)
    parser.add_argument("--network_num_actor_layers", type=int, default=2)
    parser.add_argument("--network_num_critic_layers", type=int, default=4)

    # GC
    parser.add_argument(
        "--no_only_sample_from_seen_goals",
        dest="only_sample_from_seen_goals",
        action="store_false",
    )
    parser.add_argument(
        "--no_network_sigmoid_value",
        dest="network_sigmoid_value",
        action="store_false",
    )
    parser.add_argument("--live_success_rate_decay", type=float, default=0.999)
    parser.add_argument("--goal_reward_scale", type=float, default=1.0)

    # Gridworld
    parser.add_argument("--gridworld_map_size", type=int, default=16)
    parser.add_argument("--gridworld_map_rng_seed", type=int, default=0)
    parser.add_argument("--gridworld_wall_threshold", type=float, default=0.3)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    run_ppo(args)
