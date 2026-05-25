import jax.random
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class EnvState:
    player_position: jnp.ndarray
    timestep: int


@struct.dataclass
class EnvParams:
    max_timesteps: int = 200


@struct.dataclass
class StaticEnvParams:
    map_size: int = 16
    grid_map: jnp.ndarray = (
        jax.random.uniform(jax.random.PRNGKey(0), (16, 16)) < 0.3
    ).astype(jnp.int32)


def generate_static_env_params(rng, map_size, wall_threshold):
    rng, _rng = jax.random.split(rng)
    grid_map = jax.random.uniform(_rng, (map_size, map_size)) < wall_threshold
    grid_map = grid_map.at[map_size // 2, map_size // 2].set(False)
    return StaticEnvParams(
        map_size=map_size,
        grid_map=grid_map,
    )
