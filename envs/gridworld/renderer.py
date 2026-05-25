import jax.numpy as jnp


def make_gridworld_pixel_renderer(static_env_params, gc=False):
    EMPTY = jnp.array([0, 0, 0])
    WALL = jnp.array([128, 128, 128])
    PLAYER = jnp.array([0, 255, 0])
    GOAL = jnp.array([0, 0, 255])

    def render_gridworld_pixels(state):
        pixels = jnp.where(static_env_params.grid_map[:, :, None] == 1, WALL, EMPTY)
        pixels = pixels.at[state.player_position[0], state.player_position[1]].set(
            PLAYER
        )
        return pixels

    def render_gridworld_pixels_gc(state, goal):
        pixels = jnp.where(static_env_params.grid_map[:, :, None] == 1, WALL, EMPTY)
        pixels = pixels.at[state.player_position[0], state.player_position[1]].set(
            PLAYER
        )
        pixels = pixels.at[goal[0], goal[1]].set(GOAL)
        return pixels

    if gc:
        return render_gridworld_pixels_gc
    else:
        return render_gridworld_pixels
