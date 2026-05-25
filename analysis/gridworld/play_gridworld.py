import argparse
import sys

import pygame

import jax
import jax.numpy as jnp
import numpy as np

from analysis.gridworld.pygame_renderer import KEY_MAPPING, GridworldRenderer
from envs.gridworld.gridworld_env import GridworldEnv
from envs.gridworld.gridworld_goals import sample_goal, get_all_goals, goal_achieved
from envs.gridworld.gridworld_state import generate_static_env_params


def main(args):
    static_env_params = generate_static_env_params(
        jax.random.PRNGKey(args.map_rng_seed), args.map_size, args.wall_threshold
    )
    env = GridworldEnv(static_env_params)
    env_params = env.default_params

    print("Controls")
    for k, v in KEY_MAPPING.items():
        print(f"{pygame.key.name(k)}: {v.name.lower()}")

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    _, env_state = env.reset(_rng, env_params)

    all_goals = get_all_goals(_, static_env_params)
    rng, _rng = jax.random.split(rng)
    goal_index = sample_goal(_rng, False, jnp.ones(len(all_goals)))
    goal = all_goals[goal_index]

    renderer = GridworldRenderer(
        env, env_params, static_env_params, pixel_render_size=512 // args.map_size
    )
    renderer.render(env_state, goal)

    step_fn = jax.jit(env.step)

    clock = pygame.time.Clock()

    while not renderer.is_quit_requested():
        action = renderer.get_action_from_keypress(env_state)

        if action is not None:
            rng, _rng = jax.random.split(rng)
            obs, env_state, reward, done, info = step_fn(
                _rng, env_state, action, env_params
            )

            if goal_achieved(obs, goal):
                goal_index = sample_goal(_rng, False, jnp.ones(len(all_goals)))
                goal = all_goals[goal_index]
                print("Goal achieved!")

            renderer.render(env_state, goal)

        renderer.update()
        clock.tick(args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--map_size", type=int, default=64)
    parser.add_argument("--wall_threshold", type=float, default=0.2)
    parser.add_argument("--map_rng_seed", type=int, default=0)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    main(args)
