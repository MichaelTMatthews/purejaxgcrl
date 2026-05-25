import pygame

import jax
import jax.numpy as jnp
import numpy as np

from envs.gridworld.constants import Action
from envs.gridworld.gridworld_env import GridworldEnv
from envs.gridworld.renderer import make_gridworld_pixel_renderer

KEY_MAPPING = {
    pygame.K_w: Action.UP,
    pygame.K_d: Action.RIGHT,
    pygame.K_s: Action.DOWN,
    pygame.K_a: Action.LEFT,
}


class GridworldRenderer:
    def __init__(
        self, env: GridworldEnv, env_params, static_env_params, pixel_render_size=32
    ):
        self.env = env
        self.env_params = env_params
        self.static_env_params = static_env_params
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        self.screen_size = (
            static_env_params.map_size * pixel_render_size,
            static_env_params.map_size * pixel_render_size,
        )

        # Init rendering
        pygame.init()
        pygame.key.set_repeat(250, 75)

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self.renderer = make_gridworld_pixel_renderer(self.static_env_params, gc=True)
        self._render = jax.jit(self.renderer)

    def update(self):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Update screen
        pygame.display.flip()
        # time.sleep(0.01)

    def render(self, env_state, goal):
        # Clear
        self.screen_surface.fill((0, 0, 0))

        pixels = self._render(env_state, goal)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        surface = pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, 0))

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

    def get_action_from_keypress(self, state):
        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key in KEY_MAPPING:
                    return KEY_MAPPING[event.key].value

        return None
