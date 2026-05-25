from typing import Any

import jax.numpy as jnp
from flax.training.train_state import TrainState

import os

import jax
import optax
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from models.actor_critic_gc import ActorCriticGC, ActorCriticConvSymbolicCraftaxGC
from models.pqn_models_gc import (
    QNetworkConvSymbolicCraftaxGC,
    QNetworkFlatGC,
    LEONetworkConvSymbolicCraftax,
    LEONetworkFlat,
)


def load_agent(rng, config, command, args, env, env_params, obs, all_goals):
    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = CheckpointManager(
        os.path.join(args.path, "policies"), orbax_checkpointer, options
    )

    if command in ["ppo.py", "dual_leo_ppo.py"]:
        if config["NETWORK_TYPE"] == "symbolic_flat":
            if command == "ppo.py":
                network = ActorCriticGC(
                    action_dim=env.action_space(env_params).n,
                    layer_width=config["NETWORK_LAYER_WIDTH"],
                    embedding_layers=config["NETWORK_NUM_EMBEDDING_LAYERS"],
                    actor_layers=config["NETWORK_NUM_ACTOR_LAYERS"],
                    critic_layers=config["NETWORK_NUM_CRITIC_LAYERS"],
                    sigmoid_critic=config["NETWORK_SIGMOID_VALUE"],
                )
            else:
                network = ActorCriticGC(
                    action_dim=env.action_space(env_params).n,
                    layer_width=config["NETWORK_LAYER_WIDTH"],
                    embedding_layers=config["PPO_NUM_EMBEDDING_LAYERS"],
                    actor_layers=config["PPO_NUM_ACTOR_LAYERS"],
                    critic_layers=config["PPO_NUM_CRITIC_LAYERS"],
                    sigmoid_critic=config["PPO_NETWORK_SIGMOID_VALUE"],
                )
            print("Using symbolic flat network")

        elif (
            config["NETWORK_TYPE"] == "symbolic_conv"
            and "Craftax" in config["ENV_NAME"]
        ):
            if command == "ppo.py":
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
            else:
                network = ActorCriticConvSymbolicCraftaxGC(
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
            print("Using symbolic conv network")
        else:
            raise ValueError(f"Unknown symbolic network type: {config['NETWORK_TYPE']}")

        init_x = jax.tree.map(lambda x: x[None, ...], obs)
        init_goal = jax.tree.map(lambda x: x[:1], all_goals)
        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, init_x, init_goal)
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config.get("LR", None) or config["PPO_LR"], eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        train_state = checkpoint_manager.restore(
            config["TOTAL_TIMESTEPS"], items=train_state
        )

        def _sample_action(rng, obs, goal_index):
            obs = jax.tree.map(lambda x: x[None, ...], obs)
            goal = jax.tree.map(
                lambda x: x[None, ...], jax.tree.map(lambda x: x[goal_index], all_goals)
            )

            rng, _rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, obs, goal)
            action = pi.sample(seed=_rng)
            return action[0]

        return _sample_action

    elif command == "pqn.py":
        if config["NETWORK_TYPE"] == "symbolic_conv":
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

        init_x = jax.tree.map(lambda x: x[None, ...], obs)
        init_goal = jax.tree.map(lambda x: x[:1], all_goals)

        rng, _rng = jax.random.split(rng)
        network_variables = network.init(_rng, init_x, init_goal, train=False)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

        class CustomTrainState(TrainState):
            batch_stats: Any
            timesteps: int = 0
            n_updates: int = 0
            grad_steps: int = 0

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_variables["params"],
            batch_stats=network_variables["batch_stats"],
            tx=tx,
        )

        train_state = checkpoint_manager.restore(
            config["TOTAL_TIMESTEPS"], items=train_state
        )

        def _sample_action(rng, obs, goal_index):
            q_vals = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                jax.tree.map(lambda x: x[None, ...], obs),
                jax.tree.map(
                    lambda x: x[None], jax.tree.map(lambda x: x[goal_index], all_goals)
                ),
                train=False,
            )
            return jnp.argmax(q_vals[0])

        return _sample_action

    elif command == "leo.py":
        num_goals = jax.tree.leaves(all_goals)[0].shape[0]

        if config["NETWORK_TYPE"] == "symbolic_conv":
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

        init_x = jax.tree.map(lambda x: x[None, ...], obs)

        rng, _rng = jax.random.split(rng)
        network_variables = network.init(_rng, init_x, train=False)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

        class CustomTrainState(TrainState):
            batch_stats: Any
            timesteps: int = 0
            n_updates: int = 0
            grad_steps: int = 0

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_variables["params"],
            batch_stats=network_variables["batch_stats"],
            tx=tx,
        )

        train_state = checkpoint_manager.restore(
            config["TOTAL_TIMESTEPS"], items=train_state
        )

        def _sample_action(rng, obs, goal_index):
            q_vals_all = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                jax.tree.map(lambda x: x[None, ...], obs),
                train=False,
            )
            return jnp.argmax(q_vals_all[0, goal_index])

        return _sample_action

    elif command == "dual_leo_pqn.py":
        # LEO
        leo_checkpoint_manager = CheckpointManager(
            os.path.join(args.path, "leo_policies"), orbax_checkpointer, options
        )

        num_goals = jax.tree.leaves(all_goals)[0].shape[0]

        if config["NETWORK_TYPE"] == "symbolic_conv":
            leo_network = LEONetworkConvSymbolicCraftax(
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

            uvfa_network = QNetworkConvSymbolicCraftaxGC(
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
            leo_network = LEONetworkFlat(
                action_dim=env.action_space(env_params).n,
                num_goals=num_goals,
                dense_hidden_size=config["NETWORK_LAYER_WIDTH"],
                dense_layers=config["NETWORK_DENSE_LAYERS"],
                norm_type=config["NORM_TYPE"],
                norm_input=config["NORM_INPUT"],
                normalise_output=config["NETWORK_SIGMOID_VALUE"],
            )

            uvfa_network = QNetworkFlatGC(
                action_dim=env.action_space(env_params).n,
                dense_hidden_size=config["NETWORK_LAYER_WIDTH"],
                dense_layers=config["NETWORK_DENSE_LAYERS"],
                norm_type=config["NORM_TYPE"],
                norm_input=config["NORM_INPUT"],
                sigmoid_outputs=config["NETWORK_SIGMOID_VALUE"],
            )
        else:
            raise ValueError

        init_x = jax.tree.map(lambda x: x[None, ...], obs)

        rng, _rng = jax.random.split(rng)
        network_variables = leo_network.init(_rng, init_x, train=False)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

        class CustomTrainState(TrainState):
            batch_stats: Any
            timesteps: int = 0
            n_updates: int = 0
            grad_steps: int = 0

        leo_train_state = CustomTrainState.create(
            apply_fn=leo_network.apply,
            params=network_variables["params"],
            batch_stats=network_variables["batch_stats"],
            tx=tx,
        )

        leo_train_state = leo_checkpoint_manager.restore(
            config["TOTAL_TIMESTEPS"], items=leo_train_state
        )

        # UVFA
        uvfa_checkpoint_manager = CheckpointManager(
            os.path.join(args.path, "uvfa_policies"), orbax_checkpointer, options
        )

        init_x = jax.tree.map(lambda x: x[None, ...], obs)
        init_goal = jax.tree.map(lambda x: x[:1], all_goals)

        rng, _rng = jax.random.split(rng)
        network_variables = uvfa_network.init(_rng, init_x, init_goal, train=False)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

        class CustomTrainState(TrainState):
            batch_stats: Any
            timesteps: int = 0
            n_updates: int = 0
            grad_steps: int = 0

        uvfa_train_state = CustomTrainState.create(
            apply_fn=uvfa_network.apply,
            params=network_variables["params"],
            batch_stats=network_variables["batch_stats"],
            tx=tx,
        )

        uvfa_train_state = uvfa_checkpoint_manager.restore(
            config["TOTAL_TIMESTEPS"], items=uvfa_train_state
        )

        def _sample_action(rng, obs, goal_index):
            uvfa_q_vals = uvfa_network.apply(
                {
                    "params": uvfa_train_state.params,
                    "batch_stats": uvfa_train_state.batch_stats,
                },
                jax.tree.map(lambda x: x[None, ...], obs),
                jax.tree.map(
                    lambda x: x[None], jax.tree.map(lambda x: x[goal_index], all_goals)
                ),
                train=False,
            )[0]

            leo_q_vals_all = leo_network.apply(
                {
                    "params": leo_train_state.params,
                    "batch_stats": leo_train_state.batch_stats,
                },
                jax.tree.map(lambda x: x[None, ...], obs),
                train=False,
            )
            leo_q_vals = leo_q_vals_all[0, goal_index]

            if config["DUAL_LEO_ACTING_MODE"] == "leo_act":
                acting_q_vals = leo_q_vals
            elif config["DUAL_LEO_ACTING_MODE"] == "uvfa_act":
                acting_q_vals = uvfa_q_vals
            elif config["DUAL_LEO_ACTING_MODE"] == "lc":
                if config["ANNEAL_LC_LEO"]:
                    p = leo_train_state.n_updates / config["NUM_UPDATES"]
                    coef = (
                        p * config["LC_LEO_ANNEAL_END"]
                        + (1 - p) * config["LC_LEO_ANNEAL_START"]
                    )
                else:
                    coef = config["LC_LEO_WEIGHT"]
                acting_q_vals = uvfa_q_vals * (1 - coef) + leo_q_vals * coef
            elif config["DUAL_LEO_ACTING_MODE"] == "max":
                joint_q_vals = jnp.concatenate(
                    [leo_q_vals[None], uvfa_q_vals[None]], axis=0
                )
                acting_q_vals = joint_q_vals.max(axis=0)
            elif config["DUAL_LEO_ACTING_MODE"] == "min":
                joint_q_vals = jnp.concatenate(
                    [leo_q_vals[None], uvfa_q_vals[None]], axis=0
                )
                acting_q_vals = joint_q_vals.min(axis=0)
            else:
                raise ValueError

            return jnp.argmax(acting_q_vals)

        return _sample_action

    else:
        raise ValueError(f"Unknown command {command}")
