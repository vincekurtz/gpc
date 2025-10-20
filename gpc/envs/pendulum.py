import jax
import jax.numpy as jnp
from hydrax.tasks.pendulum import Pendulum
from mujoco import mjx

from gpc.envs import TrainingEnv


class PendulumEnv(TrainingEnv):
    """Training environment for the pendulum swingup task."""

    def __init__(self, episode_length: int) -> None:
        """Set up the pendulum training environment."""
        super().__init__(
            task=Pendulum(),
            episode_length=episode_length,
        )

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        qpos = jax.random.uniform(pos_rng, (1,), minval=-jnp.pi, maxval=jnp.pi)
        qvel = jax.random.uniform(vel_rng, (1,), minval=-8.0, maxval=8.0)
        return data.replace(qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the velocity and sin/cos of the angle."""
        theta = data.qpos[0]
        theta_dot = data.qvel[0]
        return jnp.array([jnp.cos(theta), jnp.sin(theta), theta_dot])

    @property
    def observation_size(self) -> int:
        """The size of the observation space (sin, cos, theta_dot)."""
        return 3
