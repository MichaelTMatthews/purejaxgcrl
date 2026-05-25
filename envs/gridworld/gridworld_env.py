import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import spaces, environment
from typing import Tuple, Optional

from envs.gridworld.game_logic import gridworld_step
from envs.gridworld.gridworld_state import StaticEnvParams, EnvParams, EnvState
from envs.gridworld.world_gen import generate_world


class GridworldEnv(environment.Environment):
    def __init__(self, static_env_params: StaticEnvParams = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, EnvState, float, bool, dict]:
        state, reward, done = gridworld_step(
            rng, state, action, params, self.static_env_params
        )

        done = self.is_terminal(state, params)
        info = {}

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        state = generate_world(rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jax.Array:
        return state.player_position

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return state.timestep >= params.max_timesteps

    @property
    def name(self) -> str:
        return "Gridworld-v1"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            0.0,
            1.0,
            (2,),
            dtype=jnp.float32,
        )
