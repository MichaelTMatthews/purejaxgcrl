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
from models.pqn_models_gc import QNetworkConvSymbolicCraftaxGC, QNetworkFlatGC
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
    reward_gc: jnp.ndarray  # HER
    done_ep: jnp.ndarray
    done_ep_or_goal: jnp.ndarray  # HER
    done_goal: jnp.ndarray  # HER
    next_obs: jnp.ndarray
    q_val: jnp.ndarray  # HER
    goal_index: jnp.ndarray  # HER
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

    if config["HER_MODE"] == "none":
        num_her_goals_total = 1
    else:
        num_her_goals_total = 1 + config["HER_NUM_RELABELLED_GOALS"]

    all_goal_names, all_goals = get_all_goals(config["ENV_NAME"], static_env_params)
    num_goals = jax.tree.leaves(all_goals)[0].shape[0]
    print("num_goals", num_goals)

    config["TEST_NUM_GOALS"] = num_goals

    if config["HER_MODE"] == "all":
        assert config["HER_NUM_RELABELLED_GOALS"] == num_goals, (
            f"with her_mode=all {num_goals}"
        )

    config["HER_MIXED_GOAL_RANDOM_NUM"] = int(
        np.round(
            config["HER_NUM_RELABELLED_GOALS"]
            * config["HER_MIXED_GOAL_RANDOM_PROPORTION"]
        )
    )
    config["HER_MIXED_GOAL_POSITIVE_NUM"] = (
        config["HER_NUM_RELABELLED_GOALS"] - config["HER_MIXED_GOAL_RANDOM_NUM"]
    )

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
        config["NUM_ENVS"]
        * config["NUM_STEPS"]
        // config["MINIBATCH_SIZE"]
        * num_her_goals_total
    )

    print("RL num minibatches", config["NUM_MINIBATCHES"])
    print("RL minibatch size", config["MINIBATCH_SIZE"])

    assert (config["NUM_STEPS"] * config["NUM_ENVS"] * num_her_goals_total) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

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
            network = QNetworkConvSymbolicCraftaxGC(
                action_dim=env.action_space(env_params).n,
                env_name=config["ENV_NAME"],
                dense_hidden_size=config["NETWORK_LAYER_WIDTH"],
                dense_layers=config["NETWORK_DENSE_LAYERS"],
                conv_layers=config["NETWORK_CONV_LAYERS"],
                conv_features=config["NETWORK_CONV_FEATURES"],
                conv_kernel_size=config["NETWORK_CONV_KERNEL_SIZE"],
                norm_type=config["NORM_TYPE"],
                norm_input=config["NORM_INPUT"],
                sigmoid_outputs=config["NETWORK_SIGMOID_VALUE"],
            )
        elif config["NETWORK_TYPE"] == "symbolic_flat":
            network = QNetworkFlatGC(
                action_dim=env.action_space(env_params).n,
                dense_hidden_size=config["NETWORK_LAYER_WIDTH"],
                dense_layers=config["NETWORK_DENSE_LAYERS"],
                norm_type=config["NORM_TYPE"],
                norm_input=config["NORM_INPUT"],
                sigmoid_outputs=config["NETWORK_SIGMOID_VALUE"],
            )
        else:
            raise ValueError

        rng, _rng = jax.random.split(rng)
        init_x, _ = env.reset(_rng)

        example_goal = jax.tree.map(lambda x: x[0], all_goals)

        def create_agent(rng):
            rng, _rng = jax.random.split(rng)

            network_variables = network.init(
                _rng,
                init_x,
                jax.tree.map(
                    lambda x: jnp.repeat(
                        x[None, ...], repeats=config["NUM_ENVS"], axis=0
                    ),
                    example_goal,
                ),
                train=False,
            )
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

                goal_reprs = goal_indexes_to_goals(all_goals, goal_indexes)

                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    goal_reprs,
                    train=False,
                )

                # different eps for each env
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward_e, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
                )

                # GC reward
                goals_achieved = jax.vmap(goal_achieved)(new_obs, goal_reprs)

                reward_gc = goals_achieved * 1.0

                done_ep_or_goal = jnp.logical_or(new_done, goals_achieved)

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward_e=reward_e,
                    reward_gc=reward_gc,
                    done_ep=new_done,
                    done_goal=goals_achieved,
                    done_ep_or_goal=done_ep_or_goal,
                    next_obs=new_obs,
                    q_val=q_vals,
                    goal_index=goal_indexes,
                    num_goals_completed=num_goals_completed,
                )

                # Sample new goals for completed goals (including in hindsight) and terminated episodes
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
                    lambda x, y: jax.vmap(jax.lax.select)(new_done, x, y),
                    jnp.zeros_like(num_goals_completed),
                    num_goals_completed,
                )

                goal_indexes = jax.tree.map(
                    lambda x, y: jax.vmap(jax.lax.select)(done_ep_or_goal, x, y),
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

            def _relabel_trajectory(traj, goal_index_to_relabel_to):
                def _relabel_step(timestep, transition):
                    goal_repr_to_relabel_to = goal_indexes_to_goals(
                        all_goals,
                        jax.tree.map(lambda x: x[timestep], goal_index_to_relabel_to),
                    )

                    new_done_goal = goal_achieved(
                        transition.next_obs, goal_repr_to_relabel_to
                    )

                    new_done_ep_or_goal = new_done_goal | transition.done_ep

                    new_reward_gc = new_done_goal * 1.0

                    new_q_vals = network.apply(
                        {
                            "params": train_state.params,
                            "batch_stats": train_state.batch_stats,
                        },
                        jax.tree.map(lambda x: x[None, ...], transition.obs),
                        jax.tree.map(lambda x: x[None, ...], goal_repr_to_relabel_to),
                        train=False,
                    )[0]

                    transition = transition.replace(
                        goal_index=jax.tree.map(
                            lambda x: x[timestep], goal_index_to_relabel_to
                        ),
                        reward_gc=new_reward_gc,
                        done_ep_or_goal=new_done_ep_or_goal,
                        done_goal=new_done_goal,
                        q_val=new_q_vals,
                    )

                    return timestep + 1, transition

                _, relabelled_traj = jax.lax.scan(_relabel_step, 0, traj)

                return relabelled_traj

            if config["HER_MODE"] == "none":

                def _expand(t):
                    return jax.tree.map(lambda x: x[:, :, None], t)

                transitions = Transition(
                    obs=transitions.obs,
                    action=transitions.action,
                    reward_e=transitions.reward_e,
                    reward_gc=_expand(transitions.reward_gc),
                    done_ep=transitions.done_ep,
                    done_goal=_expand(transitions.done_goal),
                    done_ep_or_goal=_expand(transitions.done_ep_or_goal),
                    next_obs=transitions.next_obs,
                    q_val=_expand(transitions.q_val),
                    goal_index=_expand(transitions.goal_index),
                    num_goals_completed=transitions.num_goals_completed,
                )

            elif config["HER_MODE"] in ["random", "positive", "mixed", "all"]:
                if config["HER_MODE"] == "random":
                    rng, _rng = jax.random.split(rng)
                    _rngs = jax.random.split(
                        _rng,
                        config["NUM_STEPS"]
                        * config["NUM_ENVS"]
                        * config["HER_NUM_RELABELLED_GOALS"],
                    )
                    random_goal_indexes = jax.vmap(
                        sample_goal, in_axes=(0, None, None)
                    )(_rngs, config["ONLY_SAMPLE_FROM_SEEN_GOALS"], goals_seen)
                    relabel_goal_indexes = jax.tree.map(
                        lambda x: x.reshape(
                            (
                                config["NUM_STEPS"],
                                config["NUM_ENVS"],
                                config["HER_NUM_RELABELLED_GOALS"],
                                *x.shape[1:],
                            )
                        ),
                        random_goal_indexes,
                    )
                elif config["HER_MODE"] == "positive":
                    rng, _rng = jax.random.split(rng)
                    _rngs = jax.random.split(
                        _rng,
                        config["NUM_STEPS"]
                        * config["NUM_ENVS"]
                        * config["HER_NUM_RELABELLED_GOALS"],
                    )
                    _rngs = jax.tree.map(
                        lambda x: jnp.reshape(
                            x,
                            (
                                config["NUM_STEPS"],
                                config["NUM_ENVS"],
                                config["HER_NUM_RELABELLED_GOALS"],
                                -1,
                            ),
                        ),
                        _rngs,
                    )

                    relabel_goal_indexes = jax.vmap(
                        jax.vmap(
                            jax.vmap(
                                sample_positive_goal_index_from_obs,
                                in_axes=(0, None, None),
                            ),
                            in_axes=(0, 0, None),
                        ),
                        in_axes=(0, 0, None),
                    )(_rngs, transitions.obs, all_goals)
                elif config["HER_MODE"] == "mixed":
                    rng, _rng = jax.random.split(rng)
                    _rngs = jax.random.split(
                        _rng,
                        config["NUM_STEPS"]
                        * config["NUM_ENVS"]
                        * config["HER_MIXED_GOAL_RANDOM_NUM"],
                    )
                    random_goal_indexes = jax.vmap(
                        sample_goal, in_axes=(0, None, None)
                    )(_rngs, config["ONLY_SAMPLE_FROM_SEEN_GOALS"], goals_seen)
                    random_goal_indexes = jax.tree.map(
                        lambda x: x.reshape(
                            (
                                config["NUM_STEPS"],
                                config["NUM_ENVS"],
                                config["HER_MIXED_GOAL_RANDOM_NUM"],
                                *x.shape[1:],
                            )
                        ),
                        random_goal_indexes,
                    )

                    rng, _rng = jax.random.split(rng)
                    _rngs = jax.random.split(
                        _rng,
                        config["NUM_STEPS"]
                        * config["NUM_ENVS"]
                        * config["HER_MIXED_GOAL_POSITIVE_NUM"],
                    )
                    _rngs = jax.tree.map(
                        lambda x: jnp.reshape(
                            x,
                            (
                                config["NUM_STEPS"],
                                config["NUM_ENVS"],
                                config["HER_MIXED_GOAL_POSITIVE_NUM"],
                                -1,
                            ),
                        ),
                        _rngs,
                    )

                    positive_goal_indexes = jax.vmap(
                        jax.vmap(
                            jax.vmap(
                                sample_positive_goal_index_from_obs,
                                in_axes=(0, None, None),
                            ),
                            in_axes=(0, 0, None),
                        ),
                        in_axes=(0, 0, None),
                    )(_rngs, transitions.obs, all_goals)

                    relabel_goal_indexes = jax.tree.map(
                        lambda x, y: jnp.concatenate([x, y], axis=2),
                        random_goal_indexes,
                        positive_goal_indexes,
                    )

                elif config["HER_MODE"] == "all":
                    all_goal_indexes = jnp.arange(num_goals)

                    relabel_goal_indexes = jnp.repeat(
                        jnp.repeat(
                            all_goal_indexes[None],
                            repeats=config["NUM_ENVS"],
                            axis=0,
                        )[None],
                        repeats=config["NUM_STEPS"],
                        axis=0,
                    )
                else:
                    raise ValueError

                if config["HER_RELABEL_GRANULARITY"] == "trajectory":
                    relabel_goal_indexes = jax.tree.map(
                        lambda x: jnp.repeat(
                            x[-1:], repeats=config["NUM_STEPS"], axis=0
                        ),
                        relabel_goal_indexes,
                    )
                elif config["HER_RELABEL_GRANULARITY"] == "transition":
                    pass
                else:
                    raise ValueError

                her_trajectories = jax.vmap(
                    jax.vmap(_relabel_trajectory, in_axes=(None, 1), out_axes=1),
                    in_axes=(1, 1),
                    out_axes=1,
                )(transitions, relabel_goal_indexes)

                def _append_her(t1, t2):
                    return jax.tree.map(
                        lambda x, y: jnp.concatenate([x[:, :, None], y], axis=2), t1, t2
                    )

                transitions = transitions.replace(
                    goal_index=_append_her(
                        transitions.goal_index, her_trajectories.goal_index
                    ),
                    reward_gc=_append_her(
                        transitions.reward_gc, her_trajectories.reward_gc
                    ),
                    done_ep_or_goal=_append_her(
                        transitions.done_ep_or_goal, her_trajectories.done_ep_or_goal
                    ),
                    done_goal=_append_her(
                        transitions.done_goal, her_trajectories.done_goal
                    ),
                    q_val=_append_her(transitions.q_val, her_trajectories.q_val),
                )
            else:
                raise ValueError

            last_obs = jax.tree.map(lambda x: x[-1], transitions.next_obs)
            all_last_goal_indexes = jax.tree.map(
                lambda x: x[-1], transitions.goal_index
            )

            def _eval_last_q(_, last_goal_indexes):
                last_goal_reprs = goal_indexes_to_goals(all_goals, last_goal_indexes)

                last_q = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    last_goal_reprs,
                    train=False,
                )
                last_q = jnp.max(last_q, axis=-1)

                return None, last_q

            _, last_q = jax.lax.scan(
                _eval_last_q,
                None,
                jax.tree.map(lambda x: jnp.transpose(x, (1, 0)), all_last_goal_indexes),
            )

            last_q = jnp.transpose(last_q, (1, 0))

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.reward_gc
                    + config["GAMMA"] * (1 - transition.done_ep_or_goal) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                )
                lambda_returns = (
                    1 - transition.done_ep_or_goal
                ) * lambda_returns + transition.done_ep_or_goal * transition.reward_gc
                next_q = jnp.max(transition.q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            last_q = last_q * (1 - transitions.done_ep_or_goal[-1])
            lambda_returns = transitions.reward_gc[-1] + config["GAMMA"] * last_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree.map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_indexes):

                    train_state, rng = carry
                    # minibatch, target = minibatch_and_target

                    def _her_index(t):
                        return jax.tree.map(
                            lambda x: x[
                                minibatch_indexes[:, 0],
                                minibatch_indexes[:, 1],
                                minibatch_indexes[:, 2],
                            ],
                            t,
                        )

                    def _broadcast_index(t):
                        return jax.tree.map(
                            lambda x: x[
                                minibatch_indexes[:, 0], minibatch_indexes[:, 1]
                            ],
                            t,
                        )

                    minibatch = Transition(
                        action=_broadcast_index(transitions.action),
                        done_ep=_broadcast_index(transitions.done_ep),
                        done_ep_or_goal=_her_index(transitions.done_ep_or_goal),
                        done_goal=_her_index(transitions.done_goal),
                        goal_index=_her_index(transitions.goal_index),
                        num_goals_completed=_broadcast_index(
                            transitions.num_goals_completed
                        ),
                        next_obs=_broadcast_index(transitions.next_obs),
                        obs=_broadcast_index(transitions.obs),
                        q_val=_her_index(transitions.q_val),
                        reward_e=_broadcast_index(transitions.reward_e),
                        reward_gc=_her_index(transitions.reward_gc),
                    )

                    lambda_target = _her_index(lambda_targets)

                    def _loss_fn(params):

                        if config["Q_LAMBDA"]:
                            mb_goal_reprs = goal_indexes_to_goals(
                                all_goals, minibatch.goal_index
                            )
                            q_vals, updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.batch_stats,
                                },
                                minibatch.obs,
                                mb_goal_reprs,
                                train=True,
                                mutable=["batch_stats"],
                            )
                            target = lambda_target
                        else:
                            # if not using q_lambda, re-pass the next_obs through the network to compute target
                            all_obs = jax.tree.map(
                                lambda x, y: jnp.concatenate((x, y)),
                                minibatch.obs,
                                minibatch.next_obs,
                            )

                            all_goal_indexes = jax.tree.map(
                                lambda x, y: jnp.concatenate((x, y)),
                                minibatch.goal_index,
                                minibatch.goal_index,
                            )

                            all_goal_reprs = goal_indexes_to_goals(
                                all_goals, all_goal_indexes
                            )

                            all_q_vals, updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.batch_stats,
                                },
                                all_obs,
                                all_goal_reprs,
                                train=True,
                                mutable=["batch_stats"],
                            )
                            q_vals, q_next = jnp.split(all_q_vals, 2)
                            q_next = jax.lax.stop_gradient(q_next)
                            q_next = jnp.max(q_next, axis=-1)  # (batch_size,)
                            target = (
                                minibatch.reward_gc
                                + (1 - minibatch.done_ep_or_goal)
                                * config["GAMMA"]
                                * q_next
                            )

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
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

                step_indexes = jnp.repeat(
                    jnp.repeat(
                        jnp.arange(config["NUM_STEPS"])[:, None, None],
                        repeats=config["NUM_ENVS"],
                        axis=1,
                    ),
                    repeats=num_her_goals_total,
                    axis=2,
                )
                env_indexes = jnp.repeat(
                    jnp.repeat(
                        jnp.arange(config["NUM_ENVS"])[None, :, None],
                        repeats=config["NUM_STEPS"],
                        axis=0,
                    ),
                    repeats=num_her_goals_total,
                    axis=2,
                )
                her_indexes = jnp.repeat(
                    jnp.repeat(
                        jnp.arange(num_her_goals_total)[None, None, :],
                        repeats=config["NUM_ENVS"],
                        axis=1,
                    ),
                    repeats=config["NUM_STEPS"],
                    axis=0,
                )

                mb_indexes = jnp.concatenate(
                    [
                        step_indexes.flatten()[:, None],
                        env_indexes.flatten()[:, None],
                        her_indexes.flatten()[:, None],
                    ],
                    axis=1,
                )

                rng, _rng = jax.random.split(rng)
                mb_indexes = jax.random.permutation(_rng, mb_indexes)
                mb_indexes = jnp.reshape(
                    mb_indexes, (config["NUM_MINIBATCHES"], config["MINIBATCH_SIZE"], 3)
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), mb_indexes
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
                        transitions.done_goal[:, :, 0],
                        transitions.num_goals_completed == 0,
                    ),
                    jax.nn.one_hot(y[:, :, 0], num_classes=num_goals),
                    jnp.zeros_like(jax.nn.one_hot(y[:, :, 0], num_classes=num_goals)),
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
                    jax.nn.one_hot(y[:, :, 0], num_classes=num_goals),
                    jnp.zeros_like(jax.nn.one_hot(y[:, :, 0], num_classes=num_goals)),
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
                for goal_index, goal_name in enumerate(all_goal_names):
                    sr = live_success_rates[goal_index]
                    metrics[f"live_success_rates_{goal_name}"] = sr

                metrics["live_success_rates_aggregate/all"] = live_success_rates.mean()

                # Num goals completed
                metrics["goals/num_goals_completed"] = (
                    transitions.num_goals_completed * transitions.done_ep
                ).sum() / transitions.done_ep.sum()
                metrics["goals/goals_seen"] = goals_seen.sum()
                metrics["goals/goals_seen_ratio"] = goals_seen.sum() / num_goals

                def callback(metrics, update_step, config):
                    if metrics["update_steps"] % config["WANDB_LOG_INTERVAL"] == 0:
                        log(metrics, update_step, config)

                jax.debug.callback(
                    callback,
                    metrics,
                    train_state.n_updates,
                    config,
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

            def calc_goals_achieved(rng, flat_goals, num_envs, num_repeats):
                rng, _rng = jax.random.split(rng)
                init_obs, env_state = test_env.reset(_rng, env_params)

                flat_goals_padded = jax.tree.map(
                    lambda x: jnp.concatenate(
                        [
                            x,
                            jnp.zeros(
                                (config["TEST_NUM_ENVS"] - num_envs, *x.shape[1:])
                            ),
                        ],
                        axis=0,
                    ),
                    flat_goals,
                )

                def _env_step(carry, _):
                    env_state, last_obs, rng = carry
                    rng, _rng = jax.random.split(rng)
                    q_vals = network.apply(
                        {
                            "params": train_state.params,
                            "batch_stats": train_state.batch_stats,
                        },
                        last_obs,
                        flat_goals_padded,
                        train=False,
                    )
                    eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                    new_action = jax.vmap(eps_greedy_exploration)(
                        jax.random.split(_rng, config["TEST_NUM_ENVS"]), q_vals, eps
                    )
                    new_obs, new_env_state, reward, new_done, info = test_env.step(
                        _rng, env_state, new_action, env_params
                    )

                    goals_achieved = jax.vmap(goal_achieved)(
                        last_obs, flat_goals_padded
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
                _rng, fixed_goals, num_test_holdout_envs, config["TEST_NUM_REPEATS"]
            )

            rng, _rng = jax.random.split(rng)
            all_goals_achieved = calc_goals_achieved(_rng, all_goals, num_goals, 1)

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

        rng, _rng = jax.random.split(rng)

        goals_seen = get_goals_seen(expl_state[0], all_goals)

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


def run_pqn(config):
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

    # Core
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=int(1e6)
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
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

    # PQN
    parser.add_argument("--eps_start", type=float, default=0.2)
    parser.add_argument("--eps_finish", type=float, default=0.01)
    parser.add_argument("--eps_decay", type=float, default=0.5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--q_lambda", action="store_true")
    parser.add_argument("--lambda", type=float, default=0.0)

    # Network
    parser.add_argument(
        "--network_type",
        type=str,
        default="symbolic_conv",
        choices=["symbolic_conv", "symbolic_flat"],
    )
    parser.add_argument("--network_conv_layers", type=int, default=1)
    parser.add_argument("--network_conv_features", type=int, default=16)
    parser.add_argument("--network_conv_kernel_size", type=int, default=3)
    parser.add_argument("--no_norm_input", dest="norm_input", action="store_false")
    parser.add_argument("--norm_type", type=str, default="layer_norm")
    parser.add_argument("--network_layer_width", type=int, default=1024)
    parser.add_argument("--network_dense_layers", type=int, default=4)
    parser.add_argument(
        "--no_network_sigmoid_value",
        dest="network_sigmoid_value",
        action="store_false",
    )

    # GC
    parser.add_argument(
        "--her_mode", type=str, default="none", help="none|random|positive|mixed|all"
    )
    parser.add_argument(
        "--her_relabel_granularity",
        type=str,
        default="transition",
        help="trajectory|transition",
    )
    parser.add_argument("--her_num_relabelled_goals", type=int, default=2)
    parser.add_argument("--her_mixed_goal_random_proportion", type=float, default=0.5)
    parser.add_argument("--live_success_rate_decay", type=float, default=0.999)
    parser.add_argument(
        "--no_only_sample_from_seen_goals",
        dest="only_sample_from_seen_goals",
        action="store_false",
    )

    # Eval
    parser.add_argument("--wandb_log_interval", type=int, default=320)
    parser.add_argument("--test_during_training", action="store_true")
    parser.add_argument("--test_interval", type=float, default=0.001)
    parser.add_argument("--test_num_repeats", type=int, default=16)
    parser.add_argument("--test_num_steps", type=int, default=512)
    parser.add_argument("--eps_test", type=float, default=0.0)

    # Gridworld
    parser.add_argument("--gridworld_map_size", type=int, default=16)
    parser.add_argument("--gridworld_map_rng_seed", type=int, default=0)
    parser.add_argument("--gridworld_wall_threshold", type=float, default=0.3)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    run_pqn(args)
