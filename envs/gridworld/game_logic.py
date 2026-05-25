import jax
import jax.numpy as jnp

from envs.gridworld.constants import DIRECTIONS


def gridworld_step_lava(rng, env_state, action, env_params, static_env_params):
    next_position = env_state.player_position + DIRECTIONS[action]
    next_position = jnp.clip(
        next_position,
        jnp.zeros(2, dtype=int),
        jnp.array(
            [static_env_params.map_size - 1, static_env_params.map_size - 1],
            dtype=int,
        ),
    )

    in_lava = static_env_params.grid_map[next_position[0], next_position[1]] == 1

    env_state = env_state.replace(
        player_position=next_position,
    )

    return env_state, 0, in_lava


def gridworld_step(rng, env_state, action, env_params, static_env_params):
    maybe_next_position = env_state.player_position + DIRECTIONS[action]
    maybe_next_position = jnp.clip(
        maybe_next_position,
        jnp.zeros(2, dtype=int),
        jnp.array(
            [static_env_params.map_size - 1, static_env_params.map_size - 1],
            dtype=int,
        ),
    )

    next_position = jax.lax.select(
        static_env_params.grid_map[maybe_next_position[0], maybe_next_position[1]] == 1,
        env_state.player_position,
        maybe_next_position,
    )

    env_state = env_state.replace(
        player_position=next_position,
        timestep=env_state.timestep + 1,
    )

    done = env_state.timestep >= env_params.max_timesteps

    return env_state, 0, done
