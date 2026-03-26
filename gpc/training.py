import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from hydrax.alg_base import SamplingBasedController
from tensorboardX import SummaryWriter

from gpc.augmented import PACParams, PolicyAugmentedController
from gpc.envs import SimulatorState, TrainingEnv
from gpc.policy import Policy

Params = Any


def simulate_episode(
    env: TrainingEnv,
    ctrl: PolicyAugmentedController,
    policy: Policy,
    exploration_noise_level: float,
    rng: jax.Array,
    strategy: str = "policy",
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, SimulatorState]:
    """Starting from a random initial state, run SPC and record training data.

    Args:
        env: The training environment.
        ctrl: The sampling-based controller (augmented with a learned policy).
        policy: The generative policy network.
        exploration_noise_level: Standard deviation of the gaussian noise added
                                 to each action.
        rng: The random number generator key.
        strategy: The strategy for advancing the simulation. "policy" uses the
                  first policy sample, while "best" agregates all samples.

    Returns:
        y: The observations at each time step.
        U: The optimal actions at each time step.
        U_guess: The initial guess for the optimal actions at each time step.
        J_spc: cost of the best action sequence found by SPC at each time step.
        J_policy: cost of the best action sequence found by the policy.
        states: Vmapped simulator states at each time step.
    """
    rng, ctrl_rng, env_rng = jax.random.split(rng, 3)

    # Set the initial state of the environment
    x = env.init_state(env_rng)

    # Set the initial sampling-based controller parameters
    psi = ctrl.init_params()
    psi = psi.replace(base_params=psi.base_params.replace(rng=ctrl_rng))

    def _scan_fn(
        carry: Tuple[SimulatorState, jax.Array, PACParams], t: int
    ) -> Tuple:
        """Take simulation step, and record all data."""
        x, U, psi = carry

        # Sample action sequences from the learned policy
        # TODO: consider warm-starting the policy
        y = env._get_observation(x)
        rng, policy_rng, explore_rng = jax.random.split(psi.base_params.rng, 3)
        policy_rngs = jax.random.split(policy_rng, ctrl.num_policy_samples)
        warm_start_level = 0.0
        Us = jax.vmap(policy.apply, in_axes=(0, None, 0, None))(
            U, y, policy_rngs, warm_start_level
        )

        # Place the samples into the predictive control parameters so they
        # can be used in the predictive control update
        psi = psi.replace(
            policy_samples=Us, base_params=psi.base_params.replace(rng=rng)
        )

        # Update the action sequence with sampling-based predictive control
        psi, rollouts = ctrl.optimize(x.data, psi)
        U_star = ctrl.get_action_sequence(psi)

        # Record the lowest costs achieved by SPC and the policy. The first
        # ctrl.base_ctrl.num_samples rollouts are from SPC, while the last
        # ctrl.num_policy_samples rollouts are from the policy.
        # TODO: consider logging something more informative
        costs = jnp.sum(rollouts.costs, axis=1)
        spc_best_idx = jnp.argmin(costs[: -ctrl.num_policy_samples])
        policy_best_idx = (
            jnp.argmin(costs[-ctrl.num_policy_samples :])
            + costs.shape[0]
            - ctrl.num_policy_samples
        )
        spc_best = costs[spc_best_idx]
        policy_best = costs[policy_best_idx]

        # Step the simulation
        if strategy == "policy":
            u = Us[0, 0]
        elif strategy == "best":
            u = U_star[0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        exploration_noise = exploration_noise_level * jax.random.normal(
            explore_rng, u.shape
        )
        x = env.step(x, u + exploration_noise)

        # Record the initial guess for the optimal action sequence. This is used
        # to weigh the flow matching loss in the policy training.
        U_guess = psi.base_params.mean

        return (x, Us, psi), (y, U_star, U_guess, spc_best, policy_best, x)

    rng, u_rng = jax.random.split(rng)
    U = jax.random.normal(
        u_rng,
        (ctrl.num_policy_samples, env.task.planning_horizon, env.task.model.nu),
    )
    _, (y, U, U_guess, J_spc, J_policy, states) = jax.lax.scan(
        _scan_fn, (x, U, psi), jnp.arange(env.episode_length)
    )

    return y, U, U_guess, J_spc, J_policy, states


def fit_policy(
    observations: jax.Array,
    action_sequences: jax.Array,
    old_action_sequences: jax.Array,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch_size: int,
    num_epochs: int,
    rng: jax.Array,
    sigma_min: float = 1e-2,
) -> jax.Array:
    """Fit a flow matching model to the data.

    This model generates samples U ~ π(U|y) from the policy by flowing from
    U ~ N(0, I) to the target action sequence U*.

    Args:
        observations: The (normalized) observations y.
        action_sequences: The corresponding target action sequences U.
        old_action_sequences: The previous action sequences U_guess.
        model: The policy network, outputs the flow matching vector field.
        optimizer: The optimizer (e.g. Adam).
        batch_size: The batch size.
        num_epochs: The number of epochs.
        rng: The random number generator key.
        sigma_min: Target distribution width for flow matching, see
                   https://arxiv.org/pdf/2210.02747, eq (20-23).

    Returns:
        The loss from the last epoch.

    Note that model and optimizer are updated in-place by flax.nnx.
    """
    num_data_points = observations.shape[0]
    num_batches = max(1, num_data_points // batch_size)

    def _loss_fn(
        model: nnx.Module,
        obs: jax.Array,
        act: jax.Array,
        old_act: jax.Array,
        noise: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """Compute the flow-matching loss."""
        alpha = 1.0 - sigma_min
        noised_action = t[..., None] * act + (1 - alpha * t[..., None]) * noise
        target = act - alpha * noise
        pred = model(noised_action, obs, t)

        # Weigh the loss by how close the noise is to the old action sequence.
        # If they are similar (in terms of angle to the target action) then the
        # weight is high. Otherwise the noised sample might be approaching the
        # target action sequence from a different direction, so this sample
        # isn't so informative and we reduce the weight.
        v1 = (old_act - act).flatten()
        v2 = (noise - act).flatten()
        cosine_similarity = jnp.dot(v1, v2) / (
            jnp.linalg.norm(v1) * jnp.linalg.norm(v2) + 1e-8
        )
        weight = jax.lax.stop_gradient(jnp.exp(2 * (cosine_similarity - 1)))

        return weight * jnp.mean(jnp.square(pred - target))

    def _train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        rng: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Perform a gradient descent step on a batch of data."""
        # Get a random batch of data
        rng, batch_rng = jax.random.split(rng)
        batch_idx = jax.random.randint(
            batch_rng, (batch_size,), 0, num_data_points
        )
        batch_obs = observations[batch_idx]
        batch_act = action_sequences[batch_idx]
        batch_old_act = old_action_sequences[batch_idx]

        # Sample noise and time steps for the flow matching targets
        rng, noise_rng, t_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(noise_rng, batch_act.shape)
        t = jax.random.uniform(t_rng, (batch_size, 1))

        # Compute the loss and its gradient
        loss, grad = nnx.value_and_grad(_loss_fn)(
            model, batch_obs, batch_act, batch_old_act, noise, t
        )

        # Update the optimizer and model parameters in-place via flax.nnx
        optimizer.update(grad)

        return rng, loss

    # for i in range(num_batches * num_epochs): take a training step
    @nnx.scan
    def _scan_fn(carry: Tuple, i: int) -> Tuple:
        model, optimizer, rng = carry
        rng, loss = _train_step(model, optimizer, rng)
        return (model, optimizer, rng), loss

    _, losses = _scan_fn(
        (model, optimizer, rng), jnp.arange(num_batches * num_epochs)
    )

    return losses[-1]


def train(  # noqa: PLR0915 this is a long function, don't limit to 50 lines
    env: TrainingEnv,
    ctrl: SamplingBasedController,
    net: nnx.Module,
    num_policy_samples: int,
    log_dir: Union[Path, str],
    num_iters: int,
    num_envs: int,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    num_epochs: int = 10,
    checkpoint_every: int = 10,
    exploration_noise_level: float = 0.0,
    normalize_observations: bool = True,
    num_videos: int = 2,
    video_fps: int = 10,
    strategy: str = "policy",
) -> None:
    """Train a generative predictive controller.

    Args:
        env: The training environment.
        ctrl: The sampling-based predictive control method to use.
        net: The flow matching network architecture.
        num_policy_samples: The number of samples to draw from the policy.
        log_dir: The directory to log TensorBoard data to.
        num_iters: The number of training iterations.
        num_envs: The number of parallel environments to simulate.
        learning_rate: The learning rate for the policy network.
        batch_size: The batch size for training the policy network.
        num_epochs: The number of epochs to train the policy network.
        checkpoint_every: Number of iterations between policy checkpoint saves.
        exploration_noise_level: Standard deviation of the gaussian noise added
                                 to each action during episode simulation.
        normalize_observations: Flag for observation normalization.
        num_videos: Number of videos to render for visualization.
        video_fps: Frames per second for rendered videos.
        strategy: The strategy for choosing a control action to advance the
                  simulation during the data collection phase. "policy" uses the
                  first policy sample, while "best" agregates all samples.

    """
    rng = jax.random.key(0)

    # Check that the task has finite input bounds
    assert jnp.all(jnp.isfinite(env.task.u_min))
    assert jnp.all(jnp.isfinite(env.task.u_max))

    # Check that the sampling-based predictive controller is compatible. In
    # particular, we need access to the mean of the sampling distribution.
    _spc_params = ctrl.init_params()
    assert hasattr(
        _spc_params, "mean"
    ), f"Controller '{type(ctrl).__name__}' is not compatible with GPC."

    # Print some information about the training setup
    episode_seconds = env.episode_length * env.task.model.opt.timestep
    horizon_seconds = env.task.planning_horizon * env.task.dt
    num_samples = num_policy_samples + ctrl.num_samples
    print("Training with:")
    print(
        f"  episode length: {episode_seconds} seconds"
        f" ({env.episode_length} simulation steps)"
    )
    (
        print(
            f"  planning horizon: {horizon_seconds} seconds"
            f" ({env.task.planning_horizon} knots)"
        ),
    )
    print(
        "  Parallel rollouts per simulation step:"
        f" {num_samples * ctrl.num_randomizations * num_envs}"
        f" (= {num_samples} x {ctrl.num_randomizations} x {num_envs})"
    )
    print("")

    # Print some info about the policy architecture
    params = nnx.state(net, nnx.Param)
    total_params = sum([np.prod(x.shape) for x in jax.tree.leaves(params)], 0)
    print(f"Policy: {type(net).__name__} with {total_params} parameters")
    print("")

    # Set up the sampling-based controller and policy network
    ctrl = PolicyAugmentedController(ctrl, num_policy_samples)
    assert env.task == ctrl.task

    # Set up the policy
    normalizer = nnx.BatchNorm(
        num_features=env.observation_size,
        momentum=0.1,
        use_bias=False,
        use_scale=False,
        use_fast_variance=False,
        rngs=nnx.Rngs(0),
    )
    policy = Policy(net, normalizer, env.task.u_min, env.task.u_max)

    # Set up the optimizer
    optimizer = nnx.Optimizer(net, optax.adamw(learning_rate))

    # Set up the TensorBoard logger
    log_dir = Path(log_dir) / time.strftime("%Y%m%d_%H%M%S")
    print("Logging to", log_dir)
    tb_writer = SummaryWriter(log_dir)

    # Set up some helper functions
    @nnx.jit
    def jit_simulate(
        policy: Policy, rng: jax.Array
    ) -> Tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, SimulatorState
    ]:
        """Simulate episodes in parallel.

        Args:
            policy: The policy network.
            rng: The random number generator key.

        Returns:
            The observations at each time step.
            The best action sequence at each time step.
            Average cost of SPC's best action sequence.
            Average cost of the policy's best action sequence.
            Fraction of times the policy generated the best action sequence.
            First four simulation trajectories for visualization.
        """
        rngs = jax.random.split(rng, num_envs)

        y, U, U_guess, J_spc, J_policy, states = jax.vmap(
            simulate_episode, in_axes=(None, None, None, None, 0, None)
        )(env, ctrl, policy, exploration_noise_level, rngs, strategy)

        # Get the first few simulated trajectories
        selected_states = jax.tree.map(lambda x: x[:num_videos], states)

        frac = jnp.mean(J_policy < J_spc)
        return (
            y,
            U,
            U_guess,
            jnp.mean(J_spc),
            jnp.mean(J_policy),
            frac,
            selected_states,
        )

    @nnx.jit
    def jit_fit(
        policy: Policy,
        optimizer: nnx.Optimizer,
        observations: jax.Array,
        actions: jax.Array,
        previous_actions: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        """Fit the policy network to the data.

        Args:
            policy: The policy network (updated in place).
            optimizer: The optimizer (updated in place).
            observations: The observations.
            actions: The best action sequences.
            previous_actions: The initial/guessed action sequences.
            rng: The random number generator key.

        Returns:
            The loss from the last epoch.
        """
        # Flatten across timesteps and initial conditions
        y = observations.reshape(-1, observations.shape[-1])
        U = actions.reshape(-1, env.task.planning_horizon, env.task.model.nu)
        U_guess = previous_actions.reshape(
            -1, env.task.planning_horizon, env.task.model.nu
        )

        # Rescale the actions from [u_min, u_max] to [-1, 1]
        mean = (env.task.u_max + env.task.u_min) / 2
        scale = (env.task.u_max - env.task.u_min) / 2
        U = (U - mean) / scale
        U_guess = (U_guess - mean) / scale

        # Normalize the observations, updating the running statistics stored
        # in the policy
        y = policy.normalizer(y, use_running_average=not normalize_observations)

        # Do the regression
        return fit_policy(
            y,
            U,
            U_guess,
            policy.model,
            optimizer,
            batch_size,
            num_epochs,
            rng,
        )

    train_start = datetime.now()
    for i in range(num_iters):
        # Simulate and record the best action sequences. Some of the action
        # samples are generated via SPC and others are generated by the policy.
        policy.model.eval()
        sim_start = time.time()
        rng, episode_rng = jax.random.split(rng)
        y, U, U_guess, J_spc, J_policy, frac, traj = jit_simulate(
            policy, episode_rng
        )
        y.block_until_ready()
        sim_time = time.time() - sim_start

        # Render the first few trajectories for visualization
        # N.B. this uses CPU mujoco's rendering utils, so we need to do it
        # sequentially and outside a jit-compiled function
        if num_videos > 0:
            render_start = time.time()
            video_frames = []
            for j in range(num_videos):
                states = jax.tree.map(lambda x: x[j], traj)  # noqa: B023
                video_frames.append(env.render(states, video_fps))
            video_frames = np.stack(video_frames)
            render_time = time.time() - render_start

        # Fit the policy network U = NNet(y) to the data
        policy.model.train()
        fit_start = time.time()
        rng, fit_rng = jax.random.split(rng)
        loss = jit_fit(policy, optimizer, y, U, U_guess, fit_rng)
        loss.block_until_ready()
        fit_time = time.time() - fit_start

        # TODO: run some evaluation tests

        # Save a policy checkpoint
        if i % checkpoint_every == 0 and i > 0:
            ckpt_path = log_dir / f"policy_ckpt_{i}.pkl"
            policy.save(ckpt_path)
            print(f"Saved policy checkpoint to {ckpt_path}")

        # Print a performance summary
        time_elapsed = datetime.now() - train_start
        print(
            f"  {i+1}/{num_iters} |"
            f" policy cost {J_policy:.4f} |"
            f" spc cost {J_spc:.4f} |"
            f" {100 * frac:.2f}% policy is best |"
            f" loss {loss:.4f} |"
            f" {time_elapsed} elapsed"
        )

        # Tensorboard logging
        tb_writer.add_scalar("sim/policy_cost", J_policy, i)
        tb_writer.add_scalar("sim/spc_cost", J_spc, i)
        tb_writer.add_scalar("sim/time", sim_time, i)
        tb_writer.add_scalar("sim/policy_best_frac", frac, i)
        tb_writer.add_scalar("fit/loss", loss, i)
        tb_writer.add_scalar("fit/time", fit_time, i)
        if num_videos > 0:
            tb_writer.add_scalar("render/time", render_time, i)
            tb_writer.add_video(
                "render/trajectories", video_frames, i, fps=video_fps
            )
        tb_writer.flush()

    return policy
