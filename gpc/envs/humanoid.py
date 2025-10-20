import jax
import jax.numpy as jnp
from hydrax.tasks.humanoid_standup import HumanoidStandup
from mujoco import mjx

from gpc.envs import TrainingEnv


class HumanoidEnv(TrainingEnv):
    """Training environment for humanoid (Unitree G1) standup."""

    def __init__(self, episode_length: int) -> None:
        """Set up the walker training environment."""
        super().__init__(task=HumanoidStandup(), episode_length=episode_length)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng, ori_rng = jax.random.split(rng, 4)

        # Random positions and velocities
        qpos = self.task.qstand + 0.1 * jax.random.normal(
            pos_rng, (self.task.model.nq,)
        )
        qvel = 0.1 * jax.random.normal(vel_rng, (self.task.model.nv,))

        # Random base orientation
        u, v, w = jax.random.uniform(ori_rng, (3,))
        quat = jnp.array(
            [
                jnp.sqrt(1 - u) * jnp.sin(2 * jnp.pi * v),
                jnp.sqrt(1 - u) * jnp.cos(2 * jnp.pi * v),
                jnp.sqrt(u) * jnp.sin(2 * jnp.pi * w),
                jnp.sqrt(u) * jnp.cos(2 * jnp.pi * w),
            ]
        )
        qpos = qpos.at[3:7].set(quat)

        return data.replace(qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the full state, regularized to be agnostic to orientation."""
        height = self.task._get_torso_height(data)[None]
        orientation = self.task._get_torso_orientation(data)  # upright rotation
        return jnp.concatenate([height, orientation, data.qpos[7:], data.qvel])

    @property
    def observation_size(self) -> int:
        """The size of the observations."""
        return 68
