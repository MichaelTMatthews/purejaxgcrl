import argparse
import os
import sys
import time
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
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
from models.actor_critic_gc import ActorCriticConvSymbolicCraftaxGC, ActorCriticGC
from models.pqn_models_gc import LEONetworkConvSymbolicCraftax, LEONetworkFlat
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)


@chex.dataclass(frozen=True)
class Transition:
    obs: Any
    next_obs: Any
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_gc: jnp.ndarray
    reward_all_goals: jnp.ndarray
    done_ep: jnp.ndarray
    done_goal: jnp.ndarray
    done_all_goals: jnp.ndarray
    q_vals_all: jnp.ndarray
    goal_index: jnp.ndarray
    num_goals_completed: jnp.ndarray
    info: Any


class LeoTrainState(TrainState):
    batch_stats: Any = None
    n_updates: int = 0
    grad_steps: int = 0


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

    config["PPO_NUM_MINIBATCHES"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["PPO_MINIBATCH_SIZE"]
    )

    config["LEO_NUM_MINIBATCHES"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["LEO_MINIBATCH_SIZE"]
    )
    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "LEO_NUM_MINIBATCHES"
    ] == 0, "LEO_NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    def ppo_linear_schedule(count):
        frac = (
            1.0
            - (count // (config["PPO_NUM_MINIBATCHES"] * config["PPO_NUM_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["PPO_LR"] * frac

    leo_lr_scheduler = optax.linear_schedule(
        init_value=config["LEO_LR"],
        end_value=1e-20,
        transition_steps=config["NUM_UPDATES"]
        * config["LEO_NUM_MINIBATCHES"]
        * config["LEO_NUM_EPOCHS"],
    )

    if config["NETWORK_TYPE"] == "symbolic_conv" and "Craftax" in config["ENV_NAME"]:
        ppo_network = ActorCriticConvSymbolicCraftaxGC(
            action_dim=env.action_space(env_params).n,
            layer_width=config["NETWORK_LAYER_WIDTH"],
            conv_layers=config["PPO_CONV_LAYERS"],
            conv_features=config["PPO_CONV_FEATURES"],
            conv_kernel_size=config["PPO_CONV_KERNEL_SIZE"],
            embedding_layers=config["PPO_NUM_EMBEDDING_LAYERS"],
            actor_layers=config["PPO_NUM_ACTOR_LAYERS"],
            critic_layers=config["PPO_NUM_CRITIC_LAYERS"],
            sigmoid_critic=config["PPO_NETWORK_SIGMOID_VALUE"],
        )

        leo_network = LEONetworkConvSymbolicCraftax(
            action_dim=env.action_space(env_params).n,
            num_goals=num_goals,
            dense_hidden_size=config["NETWORK_LAYER_WIDTH"],
            dense_layers=config["LEO_DENSE_LAYERS"],
            conv_layers=config["LEO_CONV_LAYERS"],
            conv_features=config["LEO_CONV_FEATURES"],
            conv_kernel_size=config["LEO_CONV_KERNEL_SIZE"],
            norm_type=config["LEO_NORM_TYPE"],
            norm_input=config["LEO_NORM_INPUT"],
            normalise_output=config["LEO_NETWORK_SIGMOID_VALUE"],
        )
    elif config["NETWORK_TYPE"] == "symbolic_flat":
        ppo_network = ActorCriticGC(
            action_dim=env.action_space(env_params).n,
            layer_width=config["NETWORK_LAYER_WIDTH"],
            embedding_layers=config["PPO_NUM_EMBEDDING_LAYERS"],
            actor_layers=config["PPO_NUM_ACTOR_LAYERS"],
            critic_layers=config["PPO_NUM_CRITIC_LAYERS"],
            sigmoid_critic=config["PPO_NETWORK_SIGMOID_VALUE"],
        )

        leo_network = LEONetworkFlat(
            action_dim=env.action_space(env_params).n,
            num_goals=num_goals,
            dense_hidden_size=config["NETWORK_LAYER_WIDTH"],
            dense_layers=config["LEO_DENSE_LAYERS"],
            norm_type=config["LEO_NORM_TYPE"],
            norm_input=config["LEO_NORM_INPUT"],
            normalise_output=config["LEO_NETWORK_SIGMOID_VALUE"],
        )
    else:
        raise ValueError

    def train(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        init_x = jax.tree.map(lambda x: x[:1], obsv)
        example_goal = jax.tree.map(lambda x: x[0], all_goals)

        # INIT PPO
        rng, _rng = jax.random.split(rng)
        ppo_params = ppo_network.init(
            _rng, init_x, jax.tree.map(lambda x: x[None], example_goal)
        )

        if config["PPO_ANNEAL_LR"]:
            ppo_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=ppo_linear_schedule, eps=1e-5),
            )
        else:
            ppo_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["PPO_LR"], eps=1e-5),
            )
        ppo_train_state = TrainState.create(
            apply_fn=ppo_network.apply,
            params=ppo_params,
            tx=ppo_tx,
        )

        # INIT LEO
        rng, _rng = jax.random.split(rng)
        leo_variables = leo_network.init(_rng, init_x, train=False)

        leo_lr = leo_lr_scheduler if config["LEO_ANNEAL_LR"] else config["LEO_LR"]
        leo_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=leo_lr),
        )
        leo_train_state = LeoTrainState.create(
            apply_fn=leo_network.apply,
            params=leo_variables["params"],
            batch_stats=leo_variables["batch_stats"],
            tx=leo_tx,
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    ppo_ts,
                    leo_ts,
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

                goal_reprs = goal_indexes_to_goals(all_goals, goal_indexes)

                # LEO Q-values over all goals (cached for BC targets and TD target)
                q_vals_all = leo_network.apply(
                    {
                        "params": leo_ts.params,
                        "batch_stats": leo_ts.batch_stats,
                    },
                    last_obs,
                    train=False,
                )

                # PPO acts and gives its own value estimate
                rng, _rng = jax.random.split(rng)
                pi, value = ppo_network.apply(ppo_ts.params, last_obs, goal_reprs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # Step env
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                # GC reward (on acting goal) + HER over all goals
                goals_achieved_all = jax.vmap(
                    jax.vmap(goal_achieved, in_axes=(None, 0)), in_axes=(0, None)
                )(obsv, all_goals)
                done_goal = jax.vmap(goal_achieved)(obsv, goal_reprs)
                reward_gc = done_goal * config["GOAL_REWARD_SCALE"]

                info["goals_achieved"] = done_goal

                transition = Transition(
                    obs=last_obs,
                    next_obs=obsv,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward_e=reward_e,
                    reward_gc=reward_gc,
                    reward_all_goals=goals_achieved_all * 1.0,
                    done_ep=done,
                    done_goal=done_goal,
                    done_all_goals=goals_achieved_all,
                    q_vals_all=q_vals_all,
                    goal_index=goal_indexes,
                    num_goals_completed=num_goals_completed,
                    info=info,
                )

                # Sample new goals for completed goals and terminated episodes
                rng, _rng = jax.random.split(rng)
                _rngs = jax.random.split(_rng, config["NUM_ENVS"])
                new_goal_indexes = jax.vmap(sample_goal, in_axes=(0, None, None))(
                    _rngs, config["ONLY_SAMPLE_FROM_SEEN_GOALS"], goals_seen
                )

                num_goals_completed = jax.tree.map(
                    lambda x, y: jax.vmap(jax.lax.select)(done_goal, x, y),
                    num_goals_completed + 1,
                    num_goals_completed,
                )

                num_goals_completed = jax.tree.map(
                    lambda x, y: jax.vmap(jax.lax.select)(done, x, y),
                    jnp.zeros_like(num_goals_completed),
                    num_goals_completed,
                )

                done_ep_or_goal = jnp.logical_or(done, done_goal)
                goal_indexes = jax.tree.map(
                    lambda x, y: jax.vmap(jax.lax.select)(done_ep_or_goal, x, y),
                    new_goal_indexes,
                    goal_indexes,
                )

                runner_state = (
                    ppo_ts,
                    leo_ts,
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
                ppo_ts,
                leo_ts,
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
                        traj_batch.done_ep,
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

            # BOOTSTRAP: PPO last_val for GAE, and LEO q_vals_all_last for TD target
            last_goal_repr = goal_indexes_to_goals(all_goals, goal_indexes)
            _, last_val = ppo_network.apply(ppo_ts.params, last_obs, last_goal_repr)

            q_vals_all_last = leo_network.apply(
                {
                    "params": leo_ts.params,
                    "batch_stats": leo_ts.batch_stats,
                },
                last_obs,
                train=False,
            )

            # PPO GAE
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done = jax.vmap(jnp.logical_or)(
                        transition.done_ep, transition.done_goal
                    )
                    delta = (
                        transition.reward_gc
                        + config["GAMMA"] * next_value * (1 - done)
                        - transition.value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["PPO_GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # BC coefficients
            if config["ANNEAL_BC"]:
                anneal_frac = jnp.maximum(
                    0.0, 1.0 - update_step / config["NUM_UPDATES"]
                )
                bc_policy_coef_now = config["BC_COEF_POLICY"] * anneal_frac
                bc_value_coef_now = config["BC_COEF_VALUE"] * anneal_frac
            else:
                bc_policy_coef_now = config["BC_COEF_POLICY"]
                bc_value_coef_now = config["BC_COEF_VALUE"]

            # PPO UPDATE
            def _ppo_update_epoch(update_state, unused):
                def _ppo_update_minibatch(ppo_ts, batch_info):
                    traj_batch, gae, mb_targets = batch_info

                    def _loss_fn(params, traj_batch, gae, mb_targets):
                        goal_repr = goal_indexes_to_goals(
                            all_goals, traj_batch.goal_index
                        )
                        pi, value = ppo_network.apply(params, traj_batch.obs, goal_repr)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["PPO_CLIP_EPS"], config["PPO_CLIP_EPS"])
                        value_losses = jnp.square(value - mb_targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - mb_targets
                        )
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # Actor clipped surrogate
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae_norm
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["PPO_CLIP_EPS"],
                                1.0 + config["PPO_CLIP_EPS"],
                            )
                            * gae_norm
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        # BC
                        B = traj_batch.action.shape[0]
                        q_acting = traj_batch.q_vals_all[
                            jnp.arange(B), traj_batch.goal_index
                        ]
                        q_acting = jax.lax.stop_gradient(q_acting)

                        # Policy BC: pi --> argmax(LEO)
                        if config["BC_POLICY_TARGET"] == "argmax":
                            a_star = jnp.argmax(q_acting, axis=-1)
                            bc_policy_loss = -pi.log_prob(a_star).mean()
                        elif config["BC_POLICY_TARGET"] == "softmax":
                            log_target = jax.nn.log_softmax(
                                q_acting / config["BC_POLICY_TEMP"], axis=-1
                            )
                            target_probs = jnp.exp(log_target)
                            log_probs_pi = jax.nn.log_softmax(pi.logits, axis=-1)
                            bc_policy_loss = (
                                -(target_probs * log_probs_pi).sum(axis=-1).mean()
                            )
                        else:
                            raise ValueError(
                                f"Unknown BC_POLICY_TARGET {config['BC_POLICY_TARGET']}"
                            )

                        # Value BC: V --> max(LEO)
                        v_star = q_acting.max(axis=-1)
                        bc_value_loss = 0.5 * jnp.square(value - v_star).mean()

                        total_loss = (
                            loss_actor
                            + config["PPO_VF_COEF"] * value_loss
                            - config["PPO_ENT_COEF"] * entropy
                            + bc_policy_coef_now * bc_policy_loss
                            + bc_value_coef_now * bc_value_loss
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            bc_policy_loss,
                            bc_value_loss,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total, aux), grads = grad_fn(
                        ppo_ts.params, traj_batch, gae, mb_targets
                    )
                    ppo_ts = ppo_ts.apply_gradients(grads=grads)
                    return ppo_ts, (total, aux)

                (ppo_ts, traj_batch, advantages, targets, rng) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = (
                    config["PPO_MINIBATCH_SIZE"] * config["PPO_NUM_MINIBATCHES"]
                )
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
                        x, [config["PPO_NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                ppo_ts, losses = jax.lax.scan(
                    _ppo_update_minibatch, ppo_ts, minibatches
                )
                return (ppo_ts, traj_batch, advantages, targets, rng), losses

            update_state = (ppo_ts, traj_batch, advantages, targets, rng)
            update_state, ppo_loss_info = jax.lax.scan(
                _ppo_update_epoch, update_state, None, config["PPO_NUM_EPOCHS"]
            )
            ppo_ts = update_state[0]
            rng = update_state[-1]

            # LEO (PQN) UPDATE
            # max_a Q(s_{t+1}, g, a) for all goals, shape (T, B, num_goals)
            max_next_q = jnp.concatenate(
                [traj_batch.q_vals_all[1:], q_vals_all_last[None]], axis=0
            ).max(axis=-1)

            done_for_q = jnp.logical_or(
                traj_batch.done_all_goals,
                traj_batch.done_ep[:, :, None],
            )
            q_targets = traj_batch.reward_all_goals + config["GAMMA"] * max_next_q * (
                1 - done_for_q
            )

            def _leo_update_epoch(carry, _):
                leo_ts, rng = carry

                def _leo_update_minibatch(carry, minibatch_and_target):
                    leo_ts, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        q_vals, updates = leo_network.apply(
                            {
                                "params": params,
                                "batch_stats": leo_ts.batch_stats,
                            },
                            minibatch.obs,
                            train=True,
                            mutable=["batch_stats"],
                        )
                        chosen = jnp.take_along_axis(
                            q_vals, minibatch.action[:, None, None], axis=-1
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen - target).mean()
                        return loss, (updates, chosen)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(leo_ts.params)
                    leo_ts = leo_ts.apply_gradients(grads=grads)
                    leo_ts = leo_ts.replace(
                        grad_steps=leo_ts.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (leo_ts, rng), (loss, qvals)

                def preprocess(x, rng):
                    x = x.reshape(-1, *x.shape[2:])
                    x = jax.random.permutation(rng, x)
                    x = x.reshape(config["LEO_NUM_MINIBATCHES"], -1, *x.shape[1:])
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree.map(lambda x: preprocess(x, _rng), traj_batch)
                targets_q = jax.tree.map(lambda x: preprocess(x, _rng), q_targets)

                (leo_ts, rng), (loss, qvals) = jax.lax.scan(
                    _leo_update_minibatch,
                    (leo_ts, rng),
                    (minibatches, targets_q),
                )
                return (leo_ts, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (leo_ts, rng), (leo_loss, leo_qvals) = jax.lax.scan(
                _leo_update_epoch,
                (leo_ts, rng),
                None,
                config["LEO_NUM_EPOCHS"],
            )
            leo_ts = leo_ts.replace(n_updates=leo_ts.n_updates + 1)

            # METRICS
            (
                total_loss,
                (
                    value_loss,
                    a_loss,
                    entropy,
                    bc_policy_loss,
                    bc_value_loss,
                ),
            ) = ppo_loss_info
            metrics = {
                "total_loss": total_loss.mean(),
                "actor_loss": a_loss.mean(),
                "value_loss": value_loss.mean(),
                "value_loss_scaled": value_loss.mean() * config["PPO_VF_COEF"],
                "entropy_loss": entropy.mean(),
                "entropy_loss_scaled": entropy.mean() * config["PPO_ENT_COEF"],
                "bc_policy_loss": bc_policy_loss.mean(),
                "bc_policy_loss_scaled": bc_policy_loss.mean() * bc_policy_coef_now,
                "bc_value_loss": bc_value_loss.mean(),
                "bc_value_loss_scaled": bc_value_loss.mean() * bc_value_coef_now,
                "bc_policy_coef": bc_policy_coef_now,
                "bc_value_coef": bc_value_coef_now,
                "td_loss": leo_loss.mean(),
                "qvals": leo_qvals.mean(),
                "update_step": update_step,
            }

            # wandb logging
            if config["USE_WANDB"]:
                for goal_index, goal_name in enumerate(all_goal_names):
                    sr = live_success_rates[goal_index]
                    metrics[f"live_success_rates_{goal_name}"] = sr

                metrics["live_success_rates_aggregate/all"] = live_success_rates.mean()

                metrics["goals/num_goals_completed"] = (
                    traj_batch.num_goals_completed * traj_batch.done_ep
                ).sum() / traj_batch.done_ep.sum()
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
                ppo_ts,
                leo_ts,
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
            ppo_train_state,
            leo_train_state,
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


def run_dual_leo_ppo(config):
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

    if config["USE_WANDB"] and config["SAVE_POLICY"]:
        train_state = out["runner_state"][0]
        orbax_checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=True)
        path = os.path.join(wandb.run.dir, "policies")
        checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
        save_args = orbax_utils.save_args_from_target(train_state)
        checkpoint_manager.save(
            config["TOTAL_TIMESTEPS"],
            train_state,
            save_kwargs={"save_args": save_args},
        )
        print(f"saved PPO train state to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=int(1e9)
    )
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
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
    parser.add_argument("--goal_reward_scale", type=float, default=1.0)
    parser.add_argument(
        "--no_only_sample_from_seen_goals",
        dest="only_sample_from_seen_goals",
        action="store_false",
    )
    parser.add_argument("--live_success_rate_decay", type=float, default=0.999)

    # PPO
    parser.add_argument("--ppo_lr", type=float, default=2e-4)
    parser.add_argument("--ppo_num_epochs", type=int, default=1)
    parser.add_argument("--ppo_minibatch_size", type=int, default=2048)
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2)
    parser.add_argument("--ppo_ent_coef", type=float, default=0.005)
    parser.add_argument("--ppo_vf_coef", type=float, default=0.5)
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95)
    parser.add_argument(
        "--no_ppo_anneal_lr", dest="ppo_anneal_lr", action="store_false"
    )

    # PPO network
    parser.add_argument(
        "--network_type",
        type=str,
        default="symbolic_conv",
        choices=["symbolic_conv", "symbolic_flat"],
    )
    parser.add_argument("--network_layer_width", type=int, default=1024)
    parser.add_argument("--ppo_conv_layers", type=int, default=1)
    parser.add_argument("--ppo_conv_features", type=int, default=32)
    parser.add_argument("--ppo_conv_kernel_size", type=int, default=3)
    parser.add_argument("--ppo_num_embedding_layers", type=int, default=1)
    parser.add_argument("--ppo_num_actor_layers", type=int, default=2)
    parser.add_argument("--ppo_num_critic_layers", type=int, default=4)
    parser.add_argument(
        "--no_ppo_network_sigmoid_value",
        dest="ppo_network_sigmoid_value",
        action="store_false",
    )

    # LEO
    parser.add_argument("--leo_lr", type=float, default=2e-4)
    parser.add_argument("--leo_num_epochs", type=int, default=2)
    parser.add_argument("--leo_minibatch_size", type=int, default=512)
    parser.add_argument(
        "--no_leo_anneal_lr", dest="leo_anneal_lr", action="store_false"
    )
    parser.add_argument(
        "--no_leo_network_sigmoid_value",
        dest="leo_network_sigmoid_value",
        action="store_false",
    )

    # LEO network
    parser.add_argument("--leo_dense_layers", type=int, default=4)
    parser.add_argument("--leo_conv_layers", type=int, default=1)
    parser.add_argument("--leo_conv_features", type=int, default=16)
    parser.add_argument("--leo_conv_kernel_size", type=int, default=3)
    parser.add_argument(
        "--no_leo_norm_input", dest="leo_norm_input", action="store_false"
    )
    parser.add_argument("--leo_norm_type", type=str, default="layer_norm")

    # BC
    parser.add_argument("--bc_coef_policy", type=float, default=0.1)
    parser.add_argument("--bc_coef_value", type=float, default=0.0)
    parser.add_argument(
        "--bc_policy_target", type=str, default="argmax", choices=["argmax", "softmax"]
    )
    parser.add_argument("--bc_policy_temp", type=float, default=1.0)
    parser.add_argument("--anneal_bc", action="store_true")

    # Gridworld
    parser.add_argument("--gridworld_map_size", type=int, default=16)
    parser.add_argument("--gridworld_map_rng_seed", type=int, default=0)
    parser.add_argument("--gridworld_wall_threshold", type=float, default=0.3)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    run_dual_leo_ppo(args)
