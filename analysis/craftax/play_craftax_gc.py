import argparse
import sys

import pygame

import jax
import numpy as np

from craftax.craftax.constants import (
    BLOCK_PIXEL_SIZE_HUMAN,
)
from craftax.craftax_env import make_craftax_env_from_name

from analysis.craftax.pygame_renderer import KEY_MAPPING
from envs.craftax.craftax_goals import craftax_goal_set, goal_achieved
from wrappers import CraftaxOneHotDictWrapper


def main(args):
    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)

    goal_names, goals = craftax_goal_set()

    def get_goal():
        goal = None
        while goal is None:
            name = input(">")
            for i, goal_name in enumerate(goal_names):
                if goal_name == name:
                    print("Selected goal", goal_name)
                    return jax.tree.map(lambda x: x[i], goals)
            print("Invalid goal")

    goal = get_goal()

    env = make_craftax_env_from_name(args.env_name, auto_reset=False)
    env = CraftaxOneHotDictWrapper(env, args.env_name)
    env_params = env.default_params

    print("Controls")
    for k, v in KEY_MAPPING.items():
        print(f"{pygame.key.name(k)}: {v.name.lower()}")

    rng, _rng = jax.random.split(rng)
    _, env_state = env.reset(_rng, env_params)

    if args.env_name == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_classic.play_craftax_classic import CraftaxRenderer

        renderer = CraftaxRenderer(
            env, env_params, pixel_render_size=64 // BLOCK_PIXEL_SIZE_HUMAN
        )
    elif args.env_name == "Craftax-Symbolic-v1":
        from craftax.craftax.play_craftax import CraftaxRenderer

        renderer = CraftaxRenderer(
            env, env_params, pixel_render_size=64 // BLOCK_PIXEL_SIZE_HUMAN
        )
    else:
        raise ValueError

    renderer.render(env_state)

    step_fn = jax.jit(env.step, static_argnames=("params",))

    clock = pygame.time.Clock()

    while not renderer.is_quit_requested():
        action = renderer.get_action_from_keypress(env_state)

        if action is not None:
            rng, _rng = jax.random.split(rng)
            obs, env_state, reward, done, info = step_fn(
                _rng, env_state, action, env_params
            )

            if goal_achieved(obs, goal):
                print("achieved!")
                goal = get_goal()

            renderer.render(env_state)

        renderer.update()
        clock.tick(args.fps)


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)


if __name__ == "__main__":
    entry_point()
