import pygame

import jax.numpy as jnp
import numpy as np
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN
from craftax.craftax_env import make_craftax_env_from_name

from analysis.acting_agent import load_agent

import argparse
import os
import sys

import jax
import yaml

from envs.craftax.craftax_goals import get_all_goals, sample_goal, goal_achieved
from wrappers import CraftaxOneHotDictWrapper, AutoResetEnvWrapper


def main(args):
    with open(os.path.join(args.path, "config.yaml")) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {}
        for key, value in raw_config.items():
            if isinstance(value, dict) and "value" in value:
                config[key] = value["value"]

        command = list(raw_config["_wandb"]["value"]["e"].values())[0]["codePath"]

    config["NUM_ENVS"] = 1

    env = make_craftax_env_from_name(args.env_name, auto_reset=False)
    env = CraftaxOneHotDictWrapper(env, args.env_name)
    env = AutoResetEnvWrapper(env)
    env_params = env.default_params
    static_env_params = env.static_env_params

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)

    all_goal_names, all_goals = get_all_goals(args.env_name, static_env_params)
    num_goals = jax.tree.leaves(all_goals)[0].shape[0]
    rng, _rng = jax.random.split(rng)
    goal_index = sample_goal(_rng, False, jnp.ones(num_goals))
    goal = jax.tree.map(lambda x: x[goal_index], all_goals)
    print("Sampled goal", all_goal_names[goal_index])

    _agent_act_fn = load_agent(
        rng, config, command, args, env, env_params, obs, all_goals
    )

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
            if goal_achieved(obs, goal):
                print("Goal achieved", all_goal_names[goal_index])
            goal_index = sample_goal(_rng, False, jnp.ones(num_goals))
            goal = jax.tree.map(lambda x: x[goal_index], all_goals)
            print("Sampled goal", all_goal_names[goal_index])

        renderer.render(env_state)

        renderer.update()
        clock.tick(args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument("--path", type=str, required=True)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    main(args)
