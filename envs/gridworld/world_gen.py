import jax
import jax.numpy as jnp

from envs.gridworld.gridworld_state import EnvState


def generate_world(rng, env_params, static_env_params):
    rng, _rng = jax.random.split(rng)
    timestep = jax.random.randint(_rng, shape=(), minval=0, maxval=50)
    return EnvState(
        player_position=jnp.ones(2, dtype=int) * static_env_params.map_size // 2,
        timestep=timestep,
    )
