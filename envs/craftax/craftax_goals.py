from functools import partial
from typing import NamedTuple, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from craftax.craftax_classic.constants import OBS_DIM as CRAFTAX_CLASSIC_OBS_DIM
from craftax.craftax.constants import OBS_DIM as CRAFTAX_OBS_DIM, ItemType
from craftax.craftax_classic.constants import BlockType as CRAFTAX_CLASSIC_BlockType
from craftax.craftax.constants import BlockType as CRAFTAX_BlockType


class DictObservationShape(NamedTuple):
    key_order: Tuple
    dict_shape: FrozenDict


CRAFTAX_CLASSIC_OBS_KEY_ORDER = (
    "block_map",
    "mob_map",
    "player_direction",
    "intrinsics",
    "inventory",
    "light_level",
    "is_player_sleeping",
)

CRAFTAX_CLASSIC_OBS_SHAPE = DictObservationShape(
    key_order=CRAFTAX_CLASSIC_OBS_KEY_ORDER,
    dict_shape=FrozenDict(
        {
            "block_map": np.array([7, 9, 17]),
            "mob_map": np.array([7, 9, 5]),
            "player_direction": np.array([1, 4]),
            "intrinsics": np.array([4, 10]),
            "inventory": np.array([12, 10]),
            "light_level": np.array([1, 10]),
            "is_player_sleeping": np.array([1, 2]),
        }
    ),
)

CRAFTAX_OBS_KEY_ORDER = (
    "armour",
    "armour_enchantments",
    "block_map",
    "books",
    "boss_vulnerable",
    "bow",
    "bow_enchantment",
    "cleared_level",
    "dungeon_level",
    "intrinsics",
    "inventory",
    "is_player_resting",
    "is_player_sleeping",
    "item_map",
    "light_level",
    "light_map",
    "mob_map",
    "pickaxe",
    "player_direction",
    "potions",
    "spells",
    "sword",
    "sword_enchantment",
)

CRAFTAX_OBS_SHAPE = DictObservationShape(
    key_order=CRAFTAX_OBS_KEY_ORDER,
    dict_shape=FrozenDict(
        {
            "armour": np.array([4, 3]),
            "armour_enchantments": np.array([4, 3]),
            "block_map": np.array([9, 11, 37]),
            "books": np.array([1, 3]),
            "boss_vulnerable": np.array([1, 2]),
            "bow": np.array([1, 2]),
            "bow_enchantment": np.array([1, 3]),
            "cleared_level": np.array([1, 2]),
            "dungeon_level": np.array([1, 10]),
            "intrinsics": np.array([9, 20]),
            "inventory": np.array([10, 100]),
            "is_player_resting": np.array([1, 2]),
            "is_player_sleeping": np.array([1, 2]),
            "item_map": np.array([9, 11, 5]),
            "light_level": np.array([1, 10]),
            "light_map": np.array([9, 11, 2]),
            "mob_map": np.array([9, 11, 40]),
            "pickaxe": np.array([1, 5]),
            "player_direction": np.array([1, 4]),
            "potions": np.array([6, 10]),
            "spells": np.array([2, 2]),
            "sword": np.array([1, 5]),
            "sword_enchantment": np.array([1, 3]),
        }
    ),
)


def get_all_goals(env_name, _):
    if env_name == "Craftax-Classic-Symbolic-v1":
        return craftax_classic_goal_set()
    elif env_name == "Craftax-Symbolic-v1":
        return craftax_goal_set()
    else:
        raise ValueError


def env_name_to_obs_shape(env_name):
    if env_name == "Craftax-Classic-Symbolic-v1":
        return CRAFTAX_CLASSIC_OBS_SHAPE
    elif env_name == "Craftax-Symbolic-v1":
        return CRAFTAX_OBS_SHAPE
    else:
        raise ValueError


@partial(jax.jit, static_argnums=(0, 1))
def create_adjacent_goal(env_name, obs_key, direction_index, index):
    def _env_name_to_obs_dim(env_name):
        if env_name == "Craftax-Classic-Symbolic-v1":
            return CRAFTAX_CLASSIC_OBS_DIM
        elif env_name == "Craftax-Symbolic-v1":
            return CRAFTAX_OBS_DIM
        else:
            raise ValueError

    obs_shape = env_name_to_obs_shape(env_name)

    obs = jax.tree.map(lambda x: jnp.zeros(x), obs_shape.dict_shape.unfreeze())

    obs_dim = _env_name_to_obs_dim(env_name)

    direction_deltas = jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])
    centre = jnp.array([obs_dim[0] // 2, obs_dim[1] // 2])

    direction_idxs = centre[None, :] + direction_deltas
    direction_idx = direction_idxs[direction_index]

    obs[obs_key] = obs[obs_key].at[direction_idx[0], direction_idx[1], index].set(1.0)

    return obs


def goal_achieved(obs, goal):
    match = jax.tree.map(lambda x, y: x * y, obs, goal)
    match_sum = jax.tree.reduce(lambda x, y: x + jnp.sum(y), match, 0.0)

    # Union
    return match_sum > 0


@jax.jit
def sample_positive_goal_index_from_obs(rng, obs, all_goals):
    positive_goals = jax.vmap(goal_achieved, in_axes=(None, 0))(obs, all_goals)

    rng, _rng = jax.random.split(rng)
    token_idx = jax.random.choice(
        _rng, jnp.arange(len(positive_goals)), shape=(), p=positive_goals
    )

    return token_idx


def craftax_goal_set():
    obs_shape = env_name_to_obs_shape("Craftax-Symbolic-v1")
    empty_obs = jax.tree.map(lambda x: jnp.zeros(x), obs_shape.dict_shape.unfreeze())

    all_goals = jax.tree.map(lambda x: jnp.zeros((0, *x.shape)), empty_obs)

    all_goal_names = tuple()

    # Inventory
    inv_names = (
        "wood",
        "stone",
        "coal",
        "iron",
        "diamond",
        "sapphire",
        "ruby",
        "sapling",
        "torches",
        "arrows",
    )

    extended_inv_goals = [
        "wood",
        "stone",
        "coal",
        "iron",
        "torches",
        "arrows",
    ]

    for i, inv_name in enumerate(inv_names):
        for n in range(1, 10):
            goal_obs = jax.tree.map(lambda x: jnp.zeros_like(x), empty_obs)
            goal_obs["inventory"] = goal_obs["inventory"].at[i, n].set(1)
            goal = goal_obs

            all_goals = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y[None, ...]]), all_goals, goal
            )
            all_goal_names = all_goal_names + (f"inventory/{inv_name}_{n}",)

    for i, inv_name in enumerate(inv_names):
        if inv_name not in extended_inv_goals:
            continue

        for n in range(10, 100, 5):
            goal_obs = jax.tree.map(lambda x: jnp.zeros_like(x), empty_obs)
            goal_obs["inventory"] = goal_obs["inventory"].at[i, n : n + 5].set(1)
            goal = goal_obs

            all_goals = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y[None, ...]]), all_goals, goal
            )
            all_goal_names = all_goal_names + (f"inventory/{inv_name}_{n}-{n + 4}",)

    # Block Map
    valid_blocks = list(range(len(CRAFTAX_BlockType)))

    # Delete backwards to retain correct indexes
    del valid_blocks[CRAFTAX_BlockType.GRAVEL.value]
    del valid_blocks[CRAFTAX_BlockType.DARKNESS.value]
    del valid_blocks[CRAFTAX_BlockType.WOOD.value]
    del valid_blocks[CRAFTAX_BlockType.INVALID.value]
    block_names = tuple(
        [b.name for i, b in enumerate(CRAFTAX_BlockType) if i in valid_blocks]
    )
    block_idxs = jnp.array(valid_blocks)
    direction_idxs = jnp.arange(4) + 1

    direction_names = ("left", "right", "up", "down")

    block_goals = jax.vmap(
        jax.vmap(create_adjacent_goal, in_axes=(None, None, 0, None)),
        in_axes=(None, None, None, 0),
    )(
        "Craftax-Symbolic-v1",
        "block_map",
        direction_idxs,
        block_idxs,
    )

    block_goals = jax.tree.map(
        lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])), block_goals
    )

    block_full_names = []
    for block_name in block_names:
        for direction_name in direction_names:
            block_full_names.append(f"block_map/{block_name}_{direction_name}")

    block_full_names = tuple(block_full_names)

    all_goals = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y], axis=0), all_goals, block_goals
    )
    all_goal_names += block_full_names

    # Mob Map
    all_mob_names = [
        # Melee
        "zombie",
        "gnome_warrior",
        "orc_soldier",
        "lizard",
        "knight",
        "troll",
        "pigman",
        "frost_troll",
        # Passive
        "cow",
        "bat",
        "snail",
        None,
        None,
        None,
        None,
        None,
        # Ranged
        "skeleton",
        "gnome_archer",
        "orc_mage",
        "kobold",
        "knight_archer",
        "deep_thing",
        "fire_elemental",
        "ice_elemental",
        # Mob Projectile
        "arrow",
        "dagger",
        "fireball",
        None,
        "arrow2",
        "slimeball",
        "fireball2",
        "iceball2",
        # Player Projectile
        None,
        None,
        "player_fireball",
        "player_iceball",
        "player_arrow",
        None,
        None,
        None,
    ]

    mob_names = []
    mob_idxs = []

    for i, mob_name in enumerate(all_mob_names):
        if mob_name is not None:
            mob_names.append(mob_name)
            mob_idxs.append(i)

    mob_names = tuple(mob_names)
    mob_idxs = jnp.array(mob_idxs)

    mob_goals = jax.vmap(
        jax.vmap(create_adjacent_goal, in_axes=(None, None, 0, None)),
        in_axes=(None, None, None, 0),
    )(
        "Craftax-Symbolic-v1",
        "mob_map",
        direction_idxs,
        mob_idxs,
    )

    mob_goals = jax.tree.map(
        lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])), mob_goals
    )

    mob_full_names = []
    for mob_name in mob_names:
        for direction_name in direction_names:
            mob_full_names.append(f"mob_map/{mob_name}_{direction_name}")

    mob_full_names = tuple(mob_full_names)

    all_goals = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y], axis=0), all_goals, mob_goals
    )
    all_goal_names += mob_full_names

    # Item Map

    valid_items = list(range(len(ItemType)))
    del valid_items[ItemType.NONE.value]
    item_names = tuple([b.name for i, b in enumerate(ItemType) if i in valid_items])
    item_idxs = jnp.array(valid_items)

    item_goals = jax.vmap(
        jax.vmap(create_adjacent_goal, in_axes=(None, None, 0, None)),
        in_axes=(None, None, None, 0),
    )(
        "Craftax-Symbolic-v1",
        "item_map",
        direction_idxs,
        item_idxs,
    )

    item_goals = jax.tree.map(
        lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])), item_goals
    )

    item_full_names = []
    for item_name in item_names:
        for direction_name in direction_names:
            item_full_names.append(f"item_map/{item_name}_{direction_name}")

    item_full_names = tuple(item_full_names)

    all_goals = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y], axis=0), all_goals, item_goals
    )
    all_goal_names += item_full_names

    # Pickaxe and Sword
    tool_materials = ["wood", "stone", "iron", "diamond"]

    for i, tool_material in enumerate(tool_materials):
        for tool_type in ["pickaxe", "sword"]:
            goal_obs = jax.tree.map(lambda x: jnp.zeros_like(x), empty_obs)
            goal_obs[tool_type] = goal_obs[tool_type].at[0, i + 1].set(1)
            goal = goal_obs

            all_goals = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y[None, ...]]), all_goals, goal
            )
            all_goal_names = all_goal_names + (f"tools/{tool_material}_{tool_type}",)

    # Bow
    goal_obs = jax.tree.map(lambda x: jnp.zeros_like(x), empty_obs)
    goal_obs["bow"] = goal_obs["bow"].at[0, 1].set(1)
    goal = goal_obs

    all_goals = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y[None, ...]]), all_goals, goal
    )
    all_goal_names = all_goal_names + ("tools/bow",)

    # Armour
    armour_names = ["helmet", "chestplate", "pants", "boots"]

    armour_materials = ["iron", "diamond"]

    for i, armour_name in enumerate(armour_names):
        for j, armour_material in enumerate(armour_materials):
            goal_obs = jax.tree.map(lambda x: jnp.zeros_like(x), empty_obs)
            goal_obs["armour"] = goal_obs["armour"].at[i, j + 1].set(1)
            goal = goal_obs

            all_goals = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y[None, ...]]), all_goals, goal
            )
            all_goal_names = all_goal_names + (
                f"tools/{armour_material}_{armour_name}",
            )

    # Enchantments (armour, bow, sword)
    enchantment_types = ["fire", "ice"]

    for i, armour_name in enumerate(armour_names):
        for j, enchantment_type in enumerate(enchantment_types):
            goal_obs = jax.tree.map(lambda x: jnp.zeros_like(x), empty_obs)
            goal_obs["armour_enchantments"] = (
                goal_obs["armour_enchantments"].at[i, j + 1].set(1)
            )
            goal = goal_obs

            all_goals = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y[None, ...]]), all_goals, goal
            )
            all_goal_names = all_goal_names + (
                f"enchant/{armour_name}_{enchantment_type}",
            )

    for i, enchantment_type in enumerate(enchantment_types):
        for weapon_type in ["sword", "bow"]:
            goal_obs = jax.tree.map(lambda x: jnp.zeros_like(x), empty_obs)
            goal_obs[f"{weapon_type}_enchantment"] = (
                goal_obs[f"{weapon_type}_enchantment"].at[0, i + 1].set(1)
            )
            goal = goal_obs

            all_goals = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y[None, ...]]), all_goals, goal
            )
            all_goal_names = all_goal_names + (
                f"enchant/{weapon_type}_{enchantment_type}",
            )

    # Dungeon Level
    for i in range(9):
        goal_obs = jax.tree.map(lambda x: jnp.zeros_like(x), empty_obs)
        goal_obs["dungeon_level"] = goal_obs["dungeon_level"].at[0, i].set(1)
        goal = goal_obs

        all_goals = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y[None, ...]]), all_goals, goal
        )
        all_goal_names = all_goal_names + (f"dungeon_level/dlvl_{i}",)

    # Attributes
    attributes = ["dexterity", "strength", "intelligence"]
    for i, attribute in enumerate(attributes[::-1]):
        for j in range(2, 6):
            goal_obs = jax.tree.map(lambda x: jnp.zeros_like(x), empty_obs)
            goal_obs["intrinsics"] = goal_obs["intrinsics"].at[-i - 1, j].set(1)
            goal = goal_obs

            all_goals = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y[None, ...]]), all_goals, goal
            )
            all_goal_names = all_goal_names + (f"intrinsics/{attribute}_{j}",)

    return all_goal_names, all_goals


def craftax_classic_goal_set():
    # Inventory
    inv_names = ("wood", "stone", "coal", "iron", "diamond", "sapling")

    tool_names = (
        "wood_pickaxe",
        "stone_pickaxe",
        "iron_pickaxe",
        "wood_sword",
        "stone_sword",
        "iron_sword",
    )

    empty_obs = jax.tree.map(
        lambda x: jnp.zeros(x), CRAFTAX_CLASSIC_OBS_SHAPE.dict_shape.unfreeze()
    )

    inv_goals = jax.tree.map(lambda x: jnp.zeros((0, *x.shape)), empty_obs)
    inv_goal_names = []

    for i, inv_name in enumerate(inv_names):
        for n in range(1, 10):
            goal = jax.tree.map(lambda x: jnp.zeros_like(x), empty_obs)
            goal["inventory"] = goal["inventory"].at[i, n].set(1)

            inv_goals = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y[None, ...]], axis=0),
                inv_goals,
                goal,
            )
            inv_goal_names.append(f"inventory/{inv_name}_{n}")

    for i, tool_name in enumerate(tool_names):
        goal = jax.tree.map(lambda x: jnp.zeros_like(x), empty_obs)
        goal["inventory"] = goal["inventory"].at[i + len(inv_names), 1:].set(1)

        inv_goals = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y[None, ...]], axis=0),
            inv_goals,
            goal,
        )
        inv_goal_names.append(f"tools/{tool_name}")

    # Adjacent block goals
    valid_blocks = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    block_names = tuple(
        [b.name for i, b in enumerate(CRAFTAX_CLASSIC_BlockType) if i in valid_blocks]
    )
    block_idxs = jnp.array(valid_blocks)
    direction_idxs = jnp.arange(4) + 1

    direction_names = ("left", "right", "up", "down")

    block_goals = jax.vmap(
        jax.vmap(create_adjacent_goal, in_axes=(None, None, 0, None)),
        in_axes=(None, None, None, 0),
    )(
        "Craftax-Classic-Symbolic-v1",
        "block_map",
        direction_idxs,
        block_idxs,
    )

    block_goals = jax.tree.map(
        lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])), block_goals
    )

    block_full_names = []
    for block_name in block_names:
        for direction_name in direction_names:
            block_full_names.append(f"block_map/{block_name}_{direction_name}")

    block_full_names = tuple(block_full_names)

    # Adjacent mob goals
    valid_mobs = [0, 1, 2, 3]
    mob_idxs = jnp.array(valid_mobs)

    mob_goals = jax.vmap(
        jax.vmap(create_adjacent_goal, in_axes=(None, None, 0, None)),
        in_axes=(None, None, None, 0),
    )(
        "Craftax-Classic-Symbolic-v1",
        "mob_map",
        direction_idxs,
        mob_idxs,
    )

    mob_goals = jax.tree.map(
        lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])), mob_goals
    )

    mob_names = ("zombie", "cow", "skeleton", "arrow")

    mob_full_names = []
    for mob_name in mob_names:
        for direction_name in direction_names:
            mob_full_names.append(f"mob_map/{mob_name}_{direction_name}")

    mob_full_names = tuple(mob_full_names)

    all_goals = jax.tree.map(
        lambda x, y, z: jnp.concatenate([x, y, z], axis=0),
        inv_goals,
        block_goals,
        mob_goals,
    )

    all_names = tuple(inv_goal_names) + block_full_names + mob_full_names

    return all_names, all_goals


def goal_indexes_to_goals(all_goals, goal_indexes):
    return jax.tree.map(lambda x: x[goal_indexes], all_goals)


def goal_to_goal_index(all_goals, goal):
    num_goals = all_goals["block_map"].shape[0]

    target_obs_repeated = jax.tree.map(
        lambda x: jnp.repeat(x[None, ...], repeats=num_goals, axis=0),
        goal,
    )

    prod = jax.tree.map(
        lambda x, y: x * y,
        target_obs_repeated,
        all_goals,
    )

    def _reduce(t):
        return jax.tree.reduce(lambda x, y: x + y.sum(), t, jnp.asarray(0))

    reduced = jax.vmap(_reduce)(prod)

    return jnp.argmax(reduced)


def get_goals_seen(transition_batch, all_goals):
    transition_sum = jax.tree.map(
        lambda x: x.astype(bool).any(axis=0), transition_batch
    )

    now_seen_all_goals = jax.tree.map(
        lambda x, y: x * y[None, ...], all_goals, transition_sum
    )
    now_seen_flat = (
        jax.tree.reduce(lambda x, y: x + jax.vmap(jnp.sum)(y), now_seen_all_goals, 0.0)
        > 0
    )

    return now_seen_flat


def sample_goal(rng, only_sample_from_seen_goals, seen_goals):
    if only_sample_from_seen_goals:
        rng, _rng = jax.random.split(rng)
        goal_index = jax.random.choice(
            _rng,
            jnp.arange(len(seen_goals)),
            p=seen_goals,
        )
    else:
        rng, _rng = jax.random.split(rng)
        goal_index = jax.random.randint(
            _rng,
            minval=0,
            maxval=len(seen_goals),
            shape=(),
        )

    return goal_index
