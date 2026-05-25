import jax
import jax.numpy as jnp
import chex
from craftax.craftax.constants import MONSTERS_KILLED_TO_CLEAR_LEVEL
from craftax.craftax.util.game_logic_utils import is_boss_vulnerable
from flax import struct
from functools import partial
from typing import Union, Any


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert num_envs % reset_ratio == 0, (
            "Reset ratio must perfectly divide num envs."
        )
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = jax.tree.map(lambda x: x[reset_indexes], obs_re)
        state_re = jax.jax.tree.map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["episode_returns"] = new_episode_return
        info["episode_lengths"] = new_episode_length
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class CraftaxOneHotDictWrapper(GymnaxWrapper):
    """Converts the flat craftax obs to a structured dict of 1-hot observations"""

    def __init__(self, env, env_name):
        super().__init__(env)

        self.env_name = env_name

    def get_obs(self, state):
        native_obs = self._env.get_obs(state)

        if self.env_name == "Craftax-Classic-Symbolic-v1":

            def convert_to_1h_obs(obs):
                # Convert scalar observations to one-hot
                obs_1h = {}

                # Keep original one-hot observations
                for k in ["block_map", "mob_map"]:
                    obs_1h[k] = obs[k]

                # Expand player_direction
                obs_1h["player_direction"] = obs["player_direction"][..., None, :]

                # Convert sleeping to one-hot (2 values)
                obs_1h["is_player_sleeping"] = jax.nn.one_hot(
                    (obs["is_player_sleeping"] > 0.5).astype(jnp.int32), num_classes=2
                )

                # Convert other scalar values to one-hot (10 values).
                # Round rather than truncate — craftax stores these as
                # float16(value)/10, so e.g. value 4 comes back as 3.999 and
                # int() would drop it to slot 3, making the _4 goals unreachable.
                for k in ["inventory", "intrinsics"]:
                    obs_1h[k] = jax.nn.one_hot(
                        jnp.round(obs[k] * 10).astype(jnp.int32), num_classes=10
                    )

                obs_1h["light_level"] = jax.nn.one_hot(
                    (obs["light_level"] * 9.9).astype(jnp.int32), num_classes=10
                )

                return obs_1h

            all_map = jnp.reshape(native_obs[:1323], (7, 9, 21))
            mob_map = all_map[:, :, 17:]
            mob_map = jnp.concatenate(
                [mob_map, (1 - mob_map.any(axis=-1))[:, :, None]], axis=2
            )

            dict_obs = {
                "block_map": all_map[:, :, :17],
                "mob_map": mob_map,
                "inventory": native_obs[1323:1335],
                "intrinsics": native_obs[1335:1339],
                "player_direction": native_obs[1339:1343],
                "light_level": native_obs[1343:1344],
                "is_player_sleeping": native_obs[1344:1345],
            }

            return convert_to_1h_obs(dict_obs)

        elif self.env_name == "Craftax-Symbolic-v1":
            all_map = jnp.reshape(native_obs[:8217], (9, 11, 83))

            inventory_1h = jnp.array(
                [
                    jax.nn.one_hot(state.inventory.wood, num_classes=100),
                    jax.nn.one_hot(state.inventory.stone, num_classes=100),
                    jax.nn.one_hot(state.inventory.coal, num_classes=100),
                    jax.nn.one_hot(state.inventory.iron, num_classes=100),
                    jax.nn.one_hot(state.inventory.diamond, num_classes=100),
                    jax.nn.one_hot(state.inventory.sapphire, num_classes=100),
                    jax.nn.one_hot(state.inventory.ruby, num_classes=100),
                    jax.nn.one_hot(state.inventory.sapling, num_classes=100),
                    jax.nn.one_hot(state.inventory.torches, num_classes=100),
                    jax.nn.one_hot(state.inventory.arrows, num_classes=100),
                ]
            ).astype(jnp.float32)

            potions_1h = jax.nn.one_hot(state.inventory.potions, num_classes=10)
            armour_1h = jax.nn.one_hot(state.inventory.armour, num_classes=3)
            armour_enchantments_1h = jax.nn.one_hot(
                state.armour_enchantments, num_classes=3
            )

            intrinsics_1h = jnp.array(
                [
                    jax.nn.one_hot(state.player_health, num_classes=20),
                    jax.nn.one_hot(state.player_food, num_classes=20),
                    jax.nn.one_hot(state.player_drink, num_classes=20),
                    jax.nn.one_hot(state.player_energy, num_classes=20),
                    jax.nn.one_hot(state.player_mana, num_classes=20),
                    jax.nn.one_hot(state.player_xp, num_classes=20),
                    jax.nn.one_hot(state.player_dexterity, num_classes=20),
                    jax.nn.one_hot(state.player_strength, num_classes=20),
                    jax.nn.one_hot(state.player_intelligence, num_classes=20),
                ]
            ).astype(jnp.float32)

            direction = jax.nn.one_hot(state.player_direction - 1, num_classes=4)

            dict_obs = {
                "block_map": all_map[:, :, :37],
                "item_map": all_map[:, :, 37:42],
                "mob_map": all_map[:, :, 42:82],
                "light_map": jax.nn.one_hot(
                    all_map[:, :, 82].astype(jnp.int32), num_classes=2
                ),
                "inventory": inventory_1h,
                "potions": potions_1h,
                "intrinsics": intrinsics_1h,
                "player_direction": direction[None],
                "armour": armour_1h,
                "armour_enchantments": armour_enchantments_1h,
                "books": jax.nn.one_hot(state.inventory.books, num_classes=3)[None],
                "pickaxe": jax.nn.one_hot(state.inventory.pickaxe, num_classes=5)[None],
                "sword": jax.nn.one_hot(state.inventory.sword, num_classes=5)[None],
                "sword_enchantment": jax.nn.one_hot(
                    state.sword_enchantment, num_classes=3
                )[None],
                "bow_enchantment": jax.nn.one_hot(state.bow_enchantment, num_classes=3)[
                    None
                ],
                "bow": jax.nn.one_hot(state.inventory.bow, num_classes=2)[None],
                "light_level": jax.nn.one_hot(
                    (state.light_level * 9.9).astype(jnp.int32), num_classes=10
                )[None],
                "is_player_sleeping": jax.nn.one_hot(
                    jnp.asarray(state.is_sleeping > 0.5).astype(jnp.int32),
                    num_classes=2,
                )[None],
                "is_player_resting": jax.nn.one_hot(
                    jnp.asarray(state.is_resting > 0.5).astype(jnp.int32), num_classes=2
                )[None],
                "spells": jax.nn.one_hot(
                    (state.learned_spells > 0.5).astype(jnp.int32), num_classes=2
                ),
                "dungeon_level": jax.nn.one_hot(state.player_level, num_classes=10)[
                    None
                ],
                "cleared_level": jax.nn.one_hot(
                    (
                        state.monsters_killed[state.player_level]
                        >= MONSTERS_KILLED_TO_CLEAR_LEVEL
                    ).astype(jnp.int32),
                    num_classes=2,
                )[None],
                "boss_vulnerable": jax.nn.one_hot(
                    jnp.asarray(is_boss_vulnerable(state) > 0.5).astype(jnp.int32),
                    num_classes=2,
                )[None],
            }

            return dict_obs
        else:
            raise ValueError(f"Unsupported env {self.env_name}")

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        _, state = self._env.reset(key, params)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        _, state, reward, done, info = self._env.step(key, state, action, params)
        obs = self.get_obs(state)

        return obs, state, reward, done, info
