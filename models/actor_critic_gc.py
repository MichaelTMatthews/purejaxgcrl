import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

import distrax


class ActorCriticConvSymbolicCraftaxGC(nn.Module):
    action_dim: Sequence[int]
    layer_width: int

    conv_layers: int = 2
    conv_features: int = 32
    conv_kernel_size: int = 2

    embedding_layers: int = 1
    actor_layers: int = 2
    critic_layers: int = 2

    sigmoid_critic: bool = False

    @nn.compact
    def __call__(self, obs, goal):
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

        image_obs = jnp.concatenate([image_obs, image_target_obs], axis=-1)
        flat_obs = jnp.concatenate([flat_obs, flat_target_obs], axis=-1)

        image_embedding = image_obs

        # Convolutions on map
        for _ in range(self.conv_layers):
            image_embedding = nn.Conv(
                features=self.conv_features,
                kernel_size=(self.conv_kernel_size, self.conv_kernel_size),
            )(image_embedding)
            image_embedding = nn.relu(image_embedding)

        image_embedding = image_embedding.reshape(image_embedding.shape[0], -1)

        # Combine embeddings
        embedding = jnp.concatenate([image_embedding, flat_obs], axis=-1)

        for _ in range(self.embedding_layers):
            embedding = nn.Dense(
                self.layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)

        # Actor
        actor_mean = embedding
        for _ in range(self.actor_layers):
            actor_mean = nn.Dense(
                self.layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(actor_mean)
            actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic
        critic = embedding

        for _ in range(self.critic_layers):
            critic = nn.Dense(
                self.layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(critic)
            critic = nn.relu(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        if self.sigmoid_critic:
            critic = nn.sigmoid(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticGC(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    embedding_layers: int = 1
    actor_layers: int = 2
    critic_layers: int = 2
    sigmoid_critic: bool = True

    @nn.compact
    def __call__(self, obs, goal):
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
        for _ in range(self.embedding_layers):
            embedding = nn.Dense(
                self.layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)

        # Actor
        actor_mean = embedding
        for _ in range(self.actor_layers):
            actor_mean = nn.Dense(
                self.layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(actor_mean)
            actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic
        critic = embedding

        for _ in range(self.critic_layers):
            critic = nn.Dense(
                self.layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(critic)
            critic = nn.relu(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        if self.sigmoid_critic:
            critic = nn.sigmoid(critic)

        return pi, jnp.squeeze(critic, axis=-1)
