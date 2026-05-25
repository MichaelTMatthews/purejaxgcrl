import pygame

import jax.numpy as jnp
import numpy as np

from analysis.acting_agent import load_agent
from analysis.gridworld.pygame_renderer import GridworldRenderer
from envs.gridworld.gridworld_env import GridworldEnv
from envs.gridworld.gridworld_goals import sample_goal, get_all_goals, goal_achieved
from envs.gridworld.gridworld_state import generate_static_env_params

import argparse
import os
import sys

import jax
import yaml

from wrappers import AutoResetEnvWrapper


def main(args):
    with open(os.path.join(args.path, "config.yaml")) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {}
        for key, value in raw_config.items():
            if isinstance(value, dict) and "value" in value:
                config[key] = value["value"]

        command = list(raw_config["_wandb"]["value"]["e"].values())[0]["codePath"]

    config["NUM_ENVS"] = 1

    static_env_params = generate_static_env_params(
        jax.random.PRNGKey(config["GRIDWORLD_MAP_RNG_SEED"]),
        config["GRIDWORLD_MAP_SIZE"],
        config["GRIDWORLD_WALL_THRESHOLD"],
    )
    env = GridworldEnv(static_env_params)
    env = AutoResetEnvWrapper(env)
    env_params = env.default_params

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)

    all_goals = get_all_goals(None, static_env_params)
    rng, _rng = jax.random.split(rng)
    goal_index = sample_goal(_rng, False, jnp.ones(len(all_goals)))
    goal = all_goals[goal_index]

    _agent_act_fn = load_agent(
        rng, config, command, args, env, env_params, obs, all_goals
    )

    renderer = GridworldRenderer(
        env, env_params, static_env_params, pixel_render_size=32
    )
    renderer.render(env_state, goal)

    step_fn = jax.jit(env.step)

    clock = pygame.time.Clock()

    while not renderer.is_quit_requested():
        rng, _rng = jax.random.split(rng)
        action = _agent_act_fn(_rng, obs, goal_index)

        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done, info = step_fn(
            _rng, env_state, action, env_params
        )

        if (
            goal_achieved(obs, goal)
            or renderer.get_action_from_keypress(env_state) is not None
        ):
            goal_index = sample_goal(_rng, False, jnp.ones(len(all_goals)))
            goal = all_goals[goal_index]

        renderer.render(env_state, goal)

        renderer.update()
        clock.tick(args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--path", type=str, required=True)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    main(args)
