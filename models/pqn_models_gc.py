import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

from models.batch_renorm import BatchRenorm


class QNetworkConvSymbolicCraftaxGC(nn.Module):
    action_dim: Sequence[int]
    env_name: str

    dense_hidden_size: int = 512
    dense_layers: int = 4

    conv_layers: int = 2
    conv_features: int = 32
    conv_kernel_size: int = 2

    norm_type: str = "batch_norm"
    norm_input: bool = False

    sigmoid_outputs: bool = False

    @nn.compact
    def __call__(self, obs, goal, train: bool):
        def _split_obs(obs):
            batch_size = obs["block_map"].shape[0]

            image_obs = jnp.concatenate([obs[k] for k in obs if "map" in k], axis=-1)

            flat_obs = jnp.concatenate(
                [obs[k].reshape((batch_size, -1)) for k in obs if "map" not in k],
                axis=-1,
            )

            return image_obs, flat_obs

        image_obs, flat_obs = _split_obs(obs)
        image_target_obs, flat_target_obs = _split_obs(goal)

        image_embedding = jnp.concatenate([image_obs, image_target_obs], axis=-1)
        flat_embedding = jnp.concatenate(
            [
                flat_obs,
                flat_target_obs,
            ],
            axis=-1,
        )

        if self.norm_input:
            image_embedding = BatchRenorm(use_running_average=not train)(
                image_embedding
            )
            flat_embedding = BatchRenorm(use_running_average=not train)(flat_embedding)
        else:
            raise ValueError

        if self.norm_type == "layer_norm":

            def normalize(x):
                return nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":

            def normalize(x):
                return BatchRenorm(use_running_average=not train)(x)
        else:

            def normalize(x):
                return x

        # Convolutions on map
        for _ in range(self.conv_layers):
            image_embedding = nn.Conv(
                features=self.conv_features,
                kernel_size=(self.conv_kernel_size, self.conv_kernel_size),
            )(image_embedding)
            image_embedding = normalize(image_embedding)
            image_embedding = nn.relu(image_embedding)

        image_embedding = image_embedding.reshape(image_embedding.shape[0], -1)

        # Combine embeddings
        embedding = jnp.concatenate([image_embedding, flat_embedding], axis=-1)

        for _ in range(self.dense_layers):
            embedding = nn.Dense(
                self.dense_hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = normalize(embedding)
            embedding = nn.relu(embedding)

        qs = nn.Dense(
            self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(embedding)

        if self.sigmoid_outputs:
            qs = jax.nn.sigmoid(qs)

        return qs


class QNetworkFlatGC(nn.Module):
    action_dim: Sequence[int]

    dense_hidden_size: int = 512
    dense_layers: int = 4

    norm_type: str = "batch_norm"
    norm_input: bool = False

    sigmoid_outputs: bool = False

    @nn.compact
    def __call__(self, obs, goal, train: bool):
        batch_size = jax.tree.leaves(obs)[0].shape[0]

        # Flatten and concatenate obs and goal
        obs = jax.tree.map(lambda x: jnp.reshape(x, (batch_size, -1)), obs)
        obs_flat = jax.tree.reduce(
            lambda x, y: jnp.concatenate([x, y], axis=1),
            obs,
            jnp.zeros((batch_size, 0)),
        )

        goal = jax.tree.map(lambda x: jnp.reshape(x, (x.shape[0], -1)), goal)
        goal_flat = jax.tree.reduce(
            lambda x, y: jnp.concatenate([x, y], axis=1),
            goal,
            jnp.zeros((batch_size, 0)),
        )

        # Shared Embedding
        embedding = jnp.concatenate([obs_flat, goal_flat], axis=1)

        if self.norm_input:
            embedding = BatchRenorm(use_running_average=not train)(embedding)
        else:
            raise ValueError

        if self.norm_type == "layer_norm":

            def normalize(x):
                return nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":

            def normalize(x):
                return BatchRenorm(use_running_average=not train)(x)
        else:

            def normalize(x):
                return x

        for _ in range(self.dense_layers):
            embedding = nn.Dense(
                self.dense_hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = normalize(embedding)
            embedding = nn.relu(embedding)

        qs = nn.Dense(
            self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(embedding)

        if self.sigmoid_outputs:
            qs = jax.nn.sigmoid(qs)

        return qs


class LEONetworkConvSymbolicCraftax(nn.Module):
    action_dim: int
    num_goals: int

    dense_hidden_size: int = 512
    dense_layers: int = 4

    conv_layers: int = 2
    conv_features: int = 32
    conv_kernel_size: int = 2

    norm_type: str = "batch_norm"
    norm_input: bool = False

    normalise_output: bool = True

    @nn.compact
    def __call__(self, obs, train: bool):
        batch_size = obs["block_map"].shape[0]

        def _split_obs(obs):

            image_obs = jnp.concatenate([obs[k] for k in obs if "map" in k], axis=-1)

            flat_obs = jnp.concatenate(
                [obs[k].reshape((batch_size, -1)) for k in obs if "map" not in k],
                axis=-1,
            )

            return image_obs, flat_obs

        image_embedding, flat_embedding = _split_obs(obs)

        if self.norm_input:
            image_embedding = BatchRenorm(use_running_average=not train)(
                image_embedding
            )
            flat_embedding = BatchRenorm(use_running_average=not train)(flat_embedding)
        else:
            raise ValueError

        if self.norm_type == "layer_norm":

            def normalize(x):
                return nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":

            def normalize(x):
                return BatchRenorm(use_running_average=not train)(x)
        else:

            def normalize(x):
                return x

        # Convolutions on map
        for _ in range(self.conv_layers):
            image_embedding = nn.Conv(
                features=self.conv_features,
                kernel_size=(self.conv_kernel_size, self.conv_kernel_size),
            )(image_embedding)
            image_embedding = normalize(image_embedding)
            image_embedding = nn.relu(image_embedding)

        image_embedding = image_embedding.reshape(image_embedding.shape[0], -1)

        # Combine embeddings
        embedding = jnp.concatenate([image_embedding, flat_embedding], axis=-1)

        for _ in range(self.dense_layers):
            embedding = nn.Dense(
                self.dense_hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = normalize(embedding)
            embedding = nn.relu(embedding)

        qs = nn.Dense(
            self.num_goals * self.action_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(embedding)

        qs = qs.reshape((batch_size, self.num_goals, self.action_dim))

        if self.normalise_output:
            qs = jax.nn.sigmoid(qs)

        return qs


class LEONetworkFlat(nn.Module):
    action_dim: int
    num_goals: int

    dense_hidden_size: int = 512
    dense_layers: int = 4

    norm_type: str = "batch_norm"
    norm_input: bool = False

    normalise_output: bool = True

    @nn.compact
    def __call__(self, obs, train: bool):
        batch_size = obs.shape[0]

        # Flatten and concatenate obs and goal
        obs = jax.tree.map(lambda x: jnp.reshape(x, (batch_size, -1)), obs)
        embedding = jax.tree.reduce(
            lambda x, y: jnp.concatenate([x, y], axis=1),
            obs,
            jnp.zeros((batch_size, 0)),
        )

        if self.norm_input:
            embedding = BatchRenorm(use_running_average=not train)(embedding)
        else:
            raise ValueError

        if self.norm_type == "layer_norm":

            def normalize(x):
                return nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":

            def normalize(x):
                return BatchRenorm(use_running_average=not train)(x)
        else:

            def normalize(x):
                return x

        for _ in range(self.dense_layers):
            embedding = nn.Dense(
                self.dense_hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = normalize(embedding)
            embedding = nn.relu(embedding)

        qs = nn.Dense(
            self.num_goals * self.action_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(embedding)

        qs = qs.reshape((batch_size, self.num_goals, self.action_dim))

        # Q should always be between 0 and 1
        qs = jax.nn.sigmoid(qs)

        return qs
