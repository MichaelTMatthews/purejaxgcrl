import jax.random
from craftax.craftax_env import make_craftax_env_from_name

from envs.gridworld.gridworld_env import GridworldEnv
from wrappers import CraftaxOneHotDictWrapper


def create_env(config):
    if "Craftax" in config["ENV_NAME"]:
        env = make_craftax_env_from_name(
            config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
        )
        env = CraftaxOneHotDictWrapper(env, config["ENV_NAME"])
        env_params = env.default_params
        static_env_params = env.static_env_params

    elif config["ENV_NAME"] == "Gridworld-v1":
        gridworld_size = config["GRIDWORLD_MAP_SIZE"]
        from envs.gridworld.gridworld_state import generate_static_env_params

        static_env_params = generate_static_env_params(
            jax.random.PRNGKey(config["GRIDWORLD_MAP_RNG_SEED"]),
            gridworld_size,
            config["GRIDWORLD_WALL_THRESHOLD"],
        )

        env = GridworldEnv(static_env_params)
        env_params = env.default_params
    else:
        raise ValueError

    return env, env_params, static_env_params


def create_goal_set_fns(env_name):
    if "Craftax" in env_name:
        from envs.craftax.craftax_goals import (
            goal_achieved,
            goal_indexes_to_goals,
            goal_to_goal_index,
            sample_positive_goal_index_from_obs,
            get_goals_seen,
            sample_goal,
            get_all_goals,
        )

        return (
            goal_achieved,
            goal_indexes_to_goals,
            goal_to_goal_index,
            sample_positive_goal_index_from_obs,
            get_goals_seen,
            sample_goal,
            get_all_goals,
        )

    elif env_name == "Gridworld-v1":
        from envs.gridworld.gridworld_goals import (
            goal_achieved,
            goal_indexes_to_goals,
            goal_to_goal_index,
            sample_positive_goal_index_from_obs,
            get_goals_seen,
            sample_goal,
            get_all_goals_and_names,
        )

        return (
            goal_achieved,
            goal_indexes_to_goals,
            goal_to_goal_index,
            sample_positive_goal_index_from_obs,
            get_goals_seen,
            sample_goal,
            get_all_goals_and_names,
        )
    else:
        raise ValueError(f"No goal set specified for {env_name}")
