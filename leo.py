import math
import os
import sys
import time

import argparse

import chex
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

import optax
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

import wandb

from envs.envs import create_env, create_goal_set_fns
from logz.logger import log
from models.pqn_models_gc import LEONetworkConvSymbolicCraftax, LEONetworkFlat
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)


@chex.dataclass(frozen=True)
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward_e: jnp.ndarray
    reward_all_goals: jnp.ndarray
    done_ep: jnp.ndarray
    done_acting_goal: jnp.ndarray
    done_all_goals: jnp.ndarray
    next_obs: jnp.ndarray
    q_vals_all: jnp.ndarray
    goal_index: jnp.ndarray
    num_goals_completed: jnp.ndarray


class CustomTrainState(TrainState):
    batch_stats: Any = None
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config):
    basic_env, env_params, static_env_params = create_env(config)

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

    config["TEST_NUM_GOALS"] = num_goals

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    num_test_holdout_envs = config["TEST_NUM_GOALS"] * config["TEST_NUM_REPEATS"]
    num_test_all_goal_envs = (
        int(math.ceil(num_goals / config["OPTIMISTIC_RESET_RATIO"]))
        * config["OPTIMISTIC_RESET_RATIO"]
    )

    config["TEST_NUM_ENVS"] = max(num_test_holdout_envs, num_test_all_goal_envs)

    config["NUM_MINIBATCHES"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["MINIBATCH_SIZE"]
    )

    print("RL num minibatches", config["NUM_MINIBATCHES"])
    print("RL minibatch size", config["MINIBATCH_SIZE"])

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    env_params = basic_env.default_params
    log_env = LogWrapper(basic_env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
        # We do NOT wrap in an auto-reset wrapper, which means this env will continue into undefined states
        # This is fine as we only use it after a reset to test the success rate of the first episode
        test_env = BatchEnvWrapper(log_env, num_envs=config["TEST_NUM_ENVS"])
    else:
        env = BatchEnvWrapper(AutoResetEnvWrapper(log_env), num_envs=config["NUM_ENVS"])
        test_env = BatchEnvWrapper(
            AutoResetEnvWrapper(log_env), num_envs=config["TEST_NUM_ENVS"]
        )

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions

    def train(rng):
        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config["ANNEAL_LR"] else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        if (
            config["NETWORK_TYPE"] == "symbolic_conv"
            and "Craftax" in config["ENV_NAME"]
        ):
            network = LEONetworkConvSymbolicCraftax(
                action_dim=env.action_space(env_params).n,
                num_goals=num_goals,
                dense_hidden_size=config["NETWORK_LAYER_WIDTH"],
                dense_layers=config["NETWORK_DENSE_LAYERS"],
                conv_layers=config["NETWORK_CONV_LAYERS"],
                conv_features=config["NETWORK_CONV_FEATURES"],
                conv_kernel_size=config["NETWORK_CONV_KERNEL_SIZE"],
                norm_type=config["NORM_TYPE"],
                norm_input=config["NORM_INPUT"],
                normalise_output=config["NETWORK_SIGMOID_VALUE"],
            )
        elif config["NETWORK_TYPE"] == "symbolic_flat":
            network = LEONetworkFlat(
                action_dim=env.action_space(env_params).n,
                num_goals=num_goals,
                dense_hidden_size=config["NETWORK_LAYER_WIDTH"],
                dense_layers=config["NETWORK_DENSE_LAYERS"],
                norm_type=config["NORM_TYPE"],
                norm_input=config["NORM_INPUT"],
                normalise_output=config["NETWORK_SIGMOID_VALUE"],
            )
        else:
            raise ValueError

        rng, _rng = jax.random.split(rng)
        init_x, _ = env.reset(_rng)

        def create_agent(rng):
            rng, _rng = jax.random.split(rng)

            network_variables = network.init(_rng, init_x, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            (
                train_state,
                expl_state,
                test_metrics,
                rng,
                goal_indexes,
                success_counter,
                failure_counter,
                goals_seen,
                num_goals_completed,
            ) = runner_state

            live_success_rates = success_counter / (
                success_counter + failure_counter + 1e-7
            )

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng, goal_indexes, num_goals_completed = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                q_vals_all = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )

                acting_q_vals = q_vals_all[jnp.arange(config["NUM_ENVS"]), goal_indexes]

                # different eps for each env
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, acting_q_vals, eps)

                new_obs, new_env_state, reward_e, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
                )

                goals_achieved = jax.vmap(
                    jax.vmap(goal_achieved, in_axes=(None, 0)), in_axes=(0, None)
                )(new_obs, all_goals)

                acting_goals_achieved = jax.vmap(goal_achieved)(
                    new_obs, goal_indexes_to_goals(all_goals, goal_indexes)
                )

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward_e=reward_e,
                    reward_all_goals=goals_achieved * 1.0,
                    done_ep=new_done,
                    done_all_goals=goals_achieved,
                    done_acting_goal=acting_goals_achieved,
                    next_obs=new_obs,
                    q_vals_all=q_vals_all,
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
                    lambda x, y: jax.vmap(jax.lax.select)(acting_goals_achieved, x, y),
                    num_goals_completed + 1,
                    num_goals_completed,
                )

                num_goals_completed = jax.tree.map(
                    lambda x, y: jax.vmap(jax.lax.select)(new_done, x, y),
                    jnp.zeros_like(num_goals_completed),
                    num_goals_completed,
                )

                goal_indexes = jax.tree.map(
                    lambda x, y: jax.vmap(jax.lax.select)(
                        new_done | acting_goals_achieved, x, y
                    ),
                    new_goal_indexes,
                    goal_indexes,
                )

                return (
                    new_obs,
                    new_env_state,
                    rng,
                    goal_indexes,
                    num_goals_completed,
                ), (transition, info)

            # Step env
            rng, _rng = jax.random.split(rng)
            (
                (*expl_state, rng, goal_indexes, num_goals_completed),
                (
                    transitions,
                    infos,
                ),
            ) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng, goal_indexes, num_goals_completed),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"],
                n_updates=train_state.n_updates + 1,
            )  # update timesteps count

            goals_seen |= get_goals_seen(
                jax.tree.map(
                    lambda x: x.reshape((x.shape[0] * x.shape[1], *x.shape[2:])),
                    transitions.obs,
                ),
                all_goals,
            )

            last_obs = jax.tree.map(lambda x: x[-1], transitions.next_obs)
            last_q = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                last_obs,
                train=False,
            )

            max_qs_tree = jax.tree.map(
                lambda x, y: (
                    jnp.concatenate([x[1:], y[None, ...]], axis=0)
                    .max(axis=-1)
                    .reshape((config["NUM_STEPS"], config["NUM_ENVS"], -1))
                ),
                transitions.q_vals_all,
                last_q,
            )
            max_qs_list, _ = jax.tree.flatten(max_qs_tree)
            max_qs = jnp.concatenate(max_qs_list, axis=2)

            q_targets = transitions.reward_all_goals + config["GAMMA"] * max_qs * (
                1
                - jnp.logical_or(
                    transitions.done_all_goals, transitions.done_ep[:, :, None]
                )
            )

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):

                        q_vals_tree, updates = network.apply(
                            {
                                "params": params,
                                "batch_stats": train_state.batch_stats,
                            },
                            minibatch.obs,
                            train=True,
                            mutable=["batch_stats"],
                        )

                        q_vals_tree = jax.tree.map(
                            lambda x: x.reshape(
                                config["MINIBATCH_SIZE"], -1, x.shape[-1]
                            ),
                            q_vals_tree,
                        )

                        qs_list, _ = jax.tree.flatten(q_vals_tree)
                        q_vals = jnp.concatenate(qs_list, axis=1)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            minibatch.action[:, None, None],
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, (updates, chosen_action_qvals)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    x = x.reshape(
                        -1, *x.shape[2:]
                    )  # num_steps*num_envs (batch_size), ...
                    x = jax.random.permutation(rng, x)  # shuffle the transitions
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )  # num_mini_updates, batch_size/num_mini_updates, ...
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree.map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...
                targets = jax.tree.map(
                    lambda x: preprocess_transition(x, _rng), q_targets
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            metrics = {
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
            }
            done_infos = jax.tree.map(
                lambda x: (
                    (x * infos["returned_episode"]).sum()
                    / infos["returned_episode"].sum()
                ),
                infos,
            )
            metrics.update(done_infos)

            if config["TEST_DURING_TRAINING"]:
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(train_state, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({k: v for k, v in test_metrics.items()})
            else:
                test_metrics = test_metrics

            # Update success/failure counters
            successful_goals = jax.tree.map(
                lambda y: jax.vmap(jax.vmap(jax.lax.select))(
                    jnp.logical_and(
                        transitions.done_acting_goal,
                        transitions.num_goals_completed == 0,
                    ),
                    jax.nn.one_hot(y, num_classes=num_goals),
                    jnp.zeros_like(jax.nn.one_hot(y, num_classes=num_goals)),
                ),
                transitions.goal_index,
            )
            success_counter = jax.tree.map(
                lambda x, y: x + y.sum(axis=0).sum(axis=0),
                success_counter,
                successful_goals,
            )

            failed_goals = jax.tree.map(
                lambda y: jax.vmap(jax.vmap(jax.lax.select))(
                    jnp.logical_and(
                        transitions.done_ep,
                        transitions.num_goals_completed == 0,
                    ),
                    jax.nn.one_hot(y, num_classes=num_goals),
                    jnp.zeros_like(jax.nn.one_hot(y, num_classes=num_goals)),
                ),
                transitions.goal_index,
            )
            failure_counter = jax.tree.map(
                lambda x, y: x + y.sum(axis=0).sum(axis=0),
                failure_counter,
                failed_goals,
            )

            success_counter = success_counter * config["LIVE_SUCCESS_RATE_DECAY"]
            failure_counter = failure_counter * config["LIVE_SUCCESS_RATE_DECAY"]

            if config["USE_WANDB"]:
                metrics["live_success_rates_aggregate/all"] = live_success_rates.mean()

                # Num goals completed
                metrics["goals/num_goals_completed"] = (
                    transitions.num_goals_completed * transitions.done_ep
                ).sum() / transitions.done_ep.sum()
                metrics["goals/goals_seen"] = goals_seen.sum()
                metrics["goals/goals_seen_ratio"] = goals_seen.sum() / num_goals

                def callback(metrics, live_success_rates, holdout_goal_names):
                    if metrics["update_steps"] % config["WANDB_LOG_INTERVAL"] == 0:
                        for goal_index, goal_name in enumerate(holdout_goal_names):
                            sr = live_success_rates[goal_index]
                            metrics[f"live_success_rates_{goal_name}"] = sr

                        log(metrics, metrics["update_steps"], config)

                jax.debug.callback(
                    callback, metrics, live_success_rates, all_goal_names
                )

            runner_state = (
                train_state,
                tuple(expl_state),
                test_metrics,
                rng,
                goal_indexes,
                success_counter,
                failure_counter,
                goals_seen,
                num_goals_completed,
            )

            return runner_state, metrics

        def get_test_metrics(train_state, rng):

            if not config["TEST_DURING_TRAINING"]:
                return None

            fixed_goal_names, fixed_goals = all_goal_names, all_goals
            fixed_goals = jax.tree.map(
                lambda x: jnp.repeat(x, repeats=config["TEST_NUM_REPEATS"], axis=0),
                fixed_goals,
            )

            fixed_goal_indexes = jax.vmap(goal_to_goal_index, in_axes=(None, 0))(
                all_goals, fixed_goals
            )

            def calc_goals_achieved(rng, flat_goal_indexes, num_envs, num_repeats):
                rng, _rng = jax.random.split(rng)
                init_obs, env_state = test_env.reset(_rng, env_params)

                flat_goal_indexes_padded = jax.tree.map(
                    lambda x: jnp.concatenate(
                        [
                            x,
                            jnp.zeros(
                                (config["TEST_NUM_ENVS"] - num_envs, *x.shape[1:]),
                                dtype=jnp.int32,
                            ),
                        ],
                        axis=0,
                    ),
                    flat_goal_indexes,
                )

                def _env_step(carry, _):
                    env_state, last_obs, rng = carry
                    rng, _rng = jax.random.split(rng)
                    q_vals_all = network.apply(
                        {
                            "params": train_state.params,
                            "batch_stats": train_state.batch_stats,
                        },
                        last_obs,
                        train=False,
                    )

                    acting_q_vals = q_vals_all[
                        jnp.arange(config["TEST_NUM_ENVS"]),
                        flat_goal_indexes_padded.astype(jnp.int32),
                    ]

                    eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                    new_action = jax.vmap(eps_greedy_exploration)(
                        jax.random.split(_rng, config["TEST_NUM_ENVS"]),
                        acting_q_vals,
                        eps,
                    )
                    new_obs, new_env_state, reward, new_done, info = test_env.step(
                        _rng, env_state, new_action, env_params
                    )

                    goals_achieved = jax.vmap(goal_achieved)(
                        last_obs,
                        goal_indexes_to_goals(all_goals, flat_goal_indexes_padded),
                    )

                    info["goals_achieved"] = goals_achieved
                    info["done"] = new_done

                    return (new_env_state, new_obs, rng), info

                _, infos = jax.lax.scan(
                    _env_step,
                    (env_state, init_obs, _rng),
                    None,
                    config["TEST_NUM_STEPS"],
                )

                infos = jax.tree.map(lambda x: x[:, :num_envs], infos)

                # Calculate achieved goals for the FIRST episode per worker
                first_dones = jnp.argmax(infos["done"], axis=0)
                first_dones = jax.vmap(jax.lax.select)(
                    first_dones == 0,
                    jnp.ones(num_envs, dtype=jnp.int32) * config["TEST_NUM_STEPS"],
                    first_dones,
                )
                goal_mask = jnp.repeat(
                    jnp.arange(config["TEST_NUM_STEPS"])[:, None],
                    repeats=num_envs,
                    axis=1,
                )
                goal_mask = goal_mask <= first_dones[None, :]

                goals_achieved = infos["goals_achieved"] * goal_mask
                goals_achieved = goals_achieved.any(axis=0).astype(jnp.float32)

                goals_achieved = jnp.reshape(
                    goals_achieved,
                    (num_envs // num_repeats, num_repeats),
                )
                goals_achieved = goals_achieved.astype(jnp.float32).mean(axis=1)

                return goals_achieved

            rng, _rng = jax.random.split(rng)
            holdout_goals_achieved = calc_goals_achieved(
                _rng,
                fixed_goal_indexes,
                num_test_holdout_envs,
                config["TEST_NUM_REPEATS"],
            )

            rng, _rng = jax.random.split(rng)
            all_goals_achieved = calc_goals_achieved(
                _rng,
                jax.vmap(goal_to_goal_index, in_axes=(None, 0))(all_goals, all_goals),
                num_goals,
                1,
            )

            info = {}
            for i, name in enumerate(fixed_goal_names):
                info[f"holdout_goal_ratios/{name}"] = holdout_goals_achieved[i]
                info["holdout_goal_ratios/all"] = all_goals_achieved.mean()

            return info

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)

        rng, _rng = jax.random.split(rng)
        expl_state = env.reset(_rng, env_params)

        success_counter = jnp.zeros(num_goals)
        failure_counter = jnp.zeros(num_goals)

        goals_seen = get_goals_seen(expl_state[0], all_goals)

        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, config["NUM_ENVS"])
        goal_indexes = jax.vmap(sample_goal, in_axes=(0, None, None))(
            _rngs, config["ONLY_SAMPLE_FROM_SEEN_GOALS"], goals_seen
        )

        num_goals_completed = jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            expl_state,
            test_metrics,
            _rng,
            goal_indexes,
            success_counter,
            failure_counter,
            goals_seen,
            num_goals_completed,
        )

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def run_leo(config):
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

    def _save_network(rs_fn, dir_name):
        train_state = rs_fn(out["runner_state"])
        orbax_checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=True)
        path = os.path.join(wandb.run.dir, dir_name)
        checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
        print(f"saved runner state to {path}")
        save_args = orbax_utils.save_args_from_target(train_state)
        checkpoint_manager.save(
            int(config["TOTAL_TIMESTEPS"]),
            train_state,
            save_kwargs={"save_args": save_args},
        )

    if config["SAVE_POLICY"]:
        _save_network(lambda x: x[0], "policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=int(1e9)
    )  # Allow scientific notation
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--minibatch_size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
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

    # GC
    parser.add_argument("--live_success_rate_decay", type=float, default=0.999)
    parser.add_argument(
        "--no_only_sample_from_seen_goals",
        dest="only_sample_from_seen_goals",
        action="store_false",
    )

    # Network
    parser.add_argument(
        "--network_type",
        type=str,
        default="symbolic_conv",
        choices=["symbolic_conv", "symbolic_flat"],
    )
    parser.add_argument("--network_conv_layers", type=int, default=1)
    parser.add_argument("--network_conv_features", type=int, default=32)
    parser.add_argument("--network_conv_kernel_size", type=int, default=3)
    parser.add_argument("--no_norm_input", dest="norm_input", action="store_false")
    parser.add_argument("--norm_type", type=str, default="layer_norm")
    parser.add_argument("--network_layer_width", type=int, default=1024)
    parser.add_argument("--network_dense_layers", type=int, default=4)

    # Eval
    parser.add_argument("--wandb_log_interval", type=int, default=20)
    parser.add_argument("--test_during_training", action="store_true")
    parser.add_argument("--test_interval", type=float, default=0.001)
    parser.add_argument("--test_num_repeats", type=int, default=16)
    parser.add_argument("--test_num_steps", type=int, default=512)
    parser.add_argument("--eps_test", type=float, default=0.0)

    # PQN
    parser.add_argument("--eps_start", type=float, default=0.2)
    parser.add_argument("--eps_finish", type=float, default=0.01)
    parser.add_argument("--eps_decay", type=float, default=0.2)
    parser.add_argument("--num_epochs", type=int, default=2)

    parser.add_argument(
        "--no_network_sigmoid_value", dest="network_sigmoid_value", action="store_false"
    )

    # Gridworld
    parser.add_argument("--gridworld_map_size", type=int, default=16)
    parser.add_argument("--gridworld_map_rng_seed", type=int, default=0)
    parser.add_argument("--gridworld_wall_threshold", type=float, default=0.3)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    run_leo(args)
