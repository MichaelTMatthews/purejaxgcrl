from enum import Enum

import jax.numpy as jnp


class Action(Enum):
    LEFT = 0  # a
    RIGHT = 1  # d
    UP = 2  # w
    DOWN = 3  # s


DIRECTIONS = jnp.array(
    [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
    ]
)
