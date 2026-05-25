import jax
import jax.numpy as jnp


def goal_achieved(obs, goal):
    return (obs == goal).all()


def sample_positive_goal_index_from_obs(rng, obs, all_goals):
    # This is super inefficient we should just precompute the mapping but it's gridworld so who cares
    positive_goals = jax.vmap(goal_achieved, in_axes=(None, 0))(obs, all_goals)
    return jnp.argmax(positive_goals)


def get_goals_seen(obs_batch, all_goals):
    goal_idxs = jax.vmap(sample_positive_goal_index_from_obs, in_axes=(None, 0, None))(
        None, obs_batch, all_goals
    )

    seen = jnp.zeros(len(all_goals), dtype=bool)
    seen = seen.at[goal_idxs].set(True)

    return seen


def get_all_goals_and_names(_, static_env_params):
    goals = jnp.zeros((0, 2), dtype=int)
    names = []

    for x in range(static_env_params.map_size):
        for y in range(static_env_params.map_size):
            if static_env_params.grid_map[x, y] == 0:
                goals = jnp.concatenate([goals, jnp.array([[x, y]])], axis=0)

                names.append(f"position/{x}-{y}")

    return names, goals


def get_all_goals(_, static_env_params):
    _, goals = get_all_goals_and_names(_, static_env_params)

    return goals


def goal_indexes_to_goals(all_goals, goal_indexes):
    return jax.tree.map(lambda x: x[goal_indexes], all_goals)


def goal_to_goal_index(all_goals, goal):
    num_goals = all_goals.shape[0]

    target_position_repeated = jax.tree.map(
        lambda x: jnp.repeat(x[None, ...], repeats=num_goals, axis=0),
        goal,
    )

    equal = (target_position_repeated == all_goals).all(axis=-1)

    return jnp.argmax(equal)


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
