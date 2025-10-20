import jax
import jax.numpy as jnp
from hydrax.algs import PredictiveSampling
from hydrax.tasks.particle import Particle
from mujoco import mjx

from gpc.augmented import PolicyAugmentedController


def test_augmented() -> None:
    """Test the prediction-augmented controller."""
    # Task and optimizer setup
    task = Particle()
    num_knots = 10
    ps = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=0.1,
        plan_horizon=1.0,
        num_knots=num_knots,
    )
    opt = PolicyAugmentedController(ps, num_policy_samples=32)
    jit_opt = jax.jit(opt.optimize)

    # Initialize the system state and policy parameters
    state = mjx.make_data(task.model)
    state = state.replace(
        mocap_pos=state.mocap_pos.at[0, 0:2].set(jnp.array([0.01, 0.01]))
    )
    params = opt.init_params()
    params = params.replace(
        policy_samples=jnp.ones((32, num_knots, task.model.nu))
    )

    for _ in range(10):
        # Do an optimization step
        params, rollouts = jit_opt(state, params)

    # Pick the best rollout
    total_costs = jnp.sum(rollouts.costs, axis=1)
    best_idx = jnp.argmin(total_costs)
    best_ctrl = rollouts.controls[best_idx]

    assert jnp.all(best_ctrl != 0.0)
    assert jnp.all(params.policy_samples == 1.0)
    assert jnp.any(params.base_params.mean != 0.0)


if __name__ == "__main__":
    test_augmented()
