import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import pytest
from flax import nnx
from hydrax.algs import PredictiveSampling

from gpc.architectures import DenoisingMLP
from gpc.augmented import PolicyAugmentedController
from gpc.envs import ParticleEnv, SimulatorState
from gpc.policy import Policy
from gpc.training import fit_policy, simulate_episode, train


def test_simulate() -> None:
    """Test simulating an episode."""
    rng = jax.random.key(0)
    env = ParticleEnv(episode_length=13)
    num_knots = 5
    ctrl = PolicyAugmentedController(
        PredictiveSampling(
            env.task,
            num_samples=8,
            noise_level=0.1,
            plan_horizon=0.5,
            num_knots=num_knots,
        ),
        num_policy_samples=8,
    )
    net = DenoisingMLP(
        action_size=env.task.model.nu,
        observation_size=env.observation_size,
        horizon=num_knots,
        hidden_layers=[32, 32],
        rngs=nnx.Rngs(0),
    )
    normalizer = nnx.BatchNorm(
        env.observation_size,
        momentum=0.1,
        epsilon=1e-5,
        use_bias=False,
        use_scale=False,
        rngs=nnx.Rngs(0),
    )

    policy = Policy(net, normalizer, env.task.u_min, env.task.u_max)

    rng, episode_rng = jax.random.split(rng)
    y, U, U_guess, J_spc, J_policy, states = simulate_episode(
        env, ctrl, policy, 0.0, episode_rng
    )

    assert y.shape == (13, 4)
    assert U.shape == (13, 5, 2)
    assert U_guess.shape == (13, 5, 2)
    assert J_spc.shape == (13,)
    assert J_policy.shape == (13,)
    assert isinstance(states, SimulatorState)
    assert states.t.shape == (13,)
    assert states.data.qpos.shape == (13, 2)


def test_fit() -> None:
    """Test fitting the policy network."""
    rng = jax.random.key(0)

    # Make some fake data
    rng, obs_rng, act_rng = jax.random.split(rng, 3)
    y1 = jax.random.uniform(obs_rng, (64, 1))
    y2 = jax.random.uniform(obs_rng, (128, 1))
    y = jnp.concatenate([y1, y2], axis=0)
    U1 = -0.5 - y1[..., None] + 0.1 * jax.random.normal(act_rng, (64, 1, 1))
    U2 = 0.5 * y2[..., None] + 0.1 * jax.random.normal(act_rng, (128, 1, 1))
    U = jnp.concatenate([U1, U2], axis=0)

    # Plot the training data
    if __name__ == "__main__":
        plt.scatter(y, U[:, 0, 0])
        plt.xlabel("Observation")
        plt.ylabel("Action")
        plt.show(block=False)

    # Set up the policy network
    net = DenoisingMLP(
        action_size=1,
        observation_size=1,
        horizon=1,
        hidden_layers=[32, 32],
        rngs=nnx.Rngs(0),
    )

    # Set up the optimizer
    optimizer = nnx.Optimizer(net, optax.adam(1e-2))
    batch_size = 512  # can be larger than the dataset b/c added noise
    num_epochs = 1000

    # Fit the policy network
    st = time.time()
    rng, fit_rng = jax.random.split(rng)
    loss = fit_policy(y, U, U, net, optimizer, batch_size, num_epochs, fit_rng)
    print("Final loss:", loss)
    assert loss < 1.0
    print("Fit time:", time.time() - st)

    # Try generating some actions
    rng, test_rng = jax.random.split(rng)
    y_test = jnp.linspace(0.0, 1.0, 100)[:, None]
    U_test = jax.random.normal(test_rng, (100, 1, 1))
    dt = 0.1
    for t in jnp.arange(0.0, 1.0, dt):
        v = net(U_test, y_test, jnp.tile(t, (100, 1)))
        U_test += v * dt

    if __name__ == "__main__":
        plt.scatter(y_test, U_test[:, 0, 0])
        plt.xlabel("Observation")
        plt.ylabel("Action")
        plt.show()


def test_train() -> None:
    """Test the training loop."""
    log_dir = Path("_test_train")
    log_dir.mkdir(parents=True, exist_ok=True)

    env = ParticleEnv()
    num_knots = 10
    net = DenoisingMLP(
        action_size=env.task.model.nu,
        observation_size=env.observation_size,
        horizon=num_knots,
        hidden_layers=[32, 32],
        rngs=nnx.Rngs(0),
    )

    # Train with predictive sampling
    ctrl = PredictiveSampling(
        env.task,
        num_samples=8,
        noise_level=0.1,
        plan_horizon=1.0,
        num_knots=num_knots,
    )
    policy = train(
        env,
        ctrl,
        net,
        num_policy_samples=8,
        log_dir=log_dir,
        num_iters=3,
        num_envs=128,
        checkpoint_every=1,
    )

    assert isinstance(policy, Policy)

    # Test the policy
    rng = jax.random.key(0)
    y = jnp.array([-0.1, 0.1, 0.0, 0.0])
    U = jnp.zeros((num_knots, env.task.model.nu))
    U = policy.apply(U, y, rng)

    # Check that the policy output points in the right direction
    assert U.shape == (num_knots, env.task.model.nu)
    assert U[0, 0] > 0.0
    assert U[0, 1] < 0.0

    # Cleanup recursively
    shutil.rmtree(log_dir)


def test_policy() -> None:
    """Test the policy helper class."""
    rng = jax.random.key(0)
    num_steps = 5
    num_actions = 2
    num_obs = 3

    # Create a toy network
    mlp = DenoisingMLP(
        action_size=num_actions,
        observation_size=num_obs,
        horizon=num_steps,
        hidden_layers=[32, 32],
        rngs=nnx.Rngs(0),
    )

    # Create an observation normalizer
    normalizer = nnx.BatchNorm(
        num_obs,
        momentum=0.1,
        epsilon=1e-5,
        use_bias=False,
        use_scale=False,
        rngs=nnx.Rngs(0),
    )

    # Create the policy
    u_min = -2 * jnp.ones(num_actions)
    u_max = jnp.ones(num_actions)
    policy = Policy(mlp, normalizer, u_min, u_max)

    # Test running the policy
    rng, apply_rng = jax.random.split(rng)
    U = jnp.zeros((num_steps, num_actions))
    y = jnp.ones((num_obs,))
    U1 = policy.apply(U, y, apply_rng)
    assert U1.shape == (num_steps, num_actions)

    assert jnp.all(U1 != 0.0)
    assert jnp.all(U1 >= u_min)
    assert jnp.all(U1 <= u_max)

    # Save and load the policy
    local_dir = Path("_test_policy")
    local_dir.mkdir(parents=True, exist_ok=True)

    policy.save(local_dir / "policy.pkl")
    del policy

    policy2 = Policy.load(local_dir / "policy.pkl")

    U2 = jax.jit(policy2.apply)(U, y, apply_rng)
    assert jnp.allclose(U2, U1)

    # Cleanup
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_simulate()
    test_fit()
    test_train()
    test_policy()
