import argparse

import mujoco
from flax import nnx
from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive as run_sampling

from gpc.architectures import DenoisingMLP
from gpc.envs import DoubleCartPoleEnv
from gpc.policy import Policy
from gpc.sampling import BootstrappedPredictiveSampling
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Balance a double inverted pendulum on a cart"
    )
    subparsers = parser.add_subparsers(
        dest="task", help="What to do (choose one)"
    )
    subparsers.add_parser("train", help="Train (and save) a generative policy")
    subparsers.add_parser("test", help="Test a generative policy")
    subparsers.add_parser(
        "sample", help="Bootstrap sampling-based MPC with a generative policy"
    )
    args = parser.parse_args()

    # Set up the environment and save file
    env = DoubleCartPoleEnv(episode_length=400)
    save_file = "/tmp/double_cart_pole_policy.pkl"

    if args.task == "train":
        # Train the policy and save it to a file
        plan_horizon = 1.0
        num_knots = 10
        ctrl = PredictiveSampling(
            env.task,
            num_samples=16,
            noise_level=0.3,
            plan_horizon=plan_horizon,
            num_knots=num_knots,
        )
        net = DenoisingMLP(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=num_knots,
            hidden_layers=[128, 128],
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            num_policy_samples=16,
            log_dir="/tmp/gpc_double_cart_pole",
            num_iters=50,
            num_envs=256,
            num_epochs=100,
            checkpoint_every=5,
            num_videos=4,
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif args.task == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        test_interactive(env, policy, inference_timestep=0.1)

    elif args.task == "sample":
        # Use the policy to bootstrap sampling-based MPC
        policy = Policy.load(save_file)
        ctrl = BootstrappedPredictiveSampling(
            policy,
            env.get_obs,
            inference_timestep=0.01,
            num_policy_samples=4,
            task=env.task,
            num_samples=1,
            noise_level=0.3,
            plan_horizon=1.0,
            num_knots=policy.model.horizon,
        )
        mj_model = env.task.mj_model
        mj_data = mujoco.MjData(mj_model)
        run_sampling(ctrl, mj_model, mj_data, frequency=50)

    else:
        parser.print_help()
