import argparse

import mujoco
from flax import nnx
from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive as run_sampling

from gpc.architectures import DenoisingCNN
from gpc.envs import WalkerEnv
from gpc.policy import Policy
from gpc.sampling import BootstrappedPredictiveSampling
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Planar biped locomotion")
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
    env = WalkerEnv(episode_length=500)
    save_file = "/tmp/walker_policy.pkl"

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
        net = DenoisingCNN(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=num_knots,
            feature_dims=[64, 64],
            timestep_embedding_dim=16,
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            log_dir="/tmp/gpc_walker",
            num_policy_samples=16,
            num_iters=20,
            num_envs=128,
            num_epochs=10,
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif args.task == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        test_interactive(
            env, policy, inference_timestep=0.01, warm_start_level=1.0
        )

    elif args.task == "sample":
        # Use the policy to bootstrap sampling-based MPC
        policy = Policy.load(save_file)
        ctrl = BootstrappedPredictiveSampling(
            policy,
            env.get_obs,
            warm_start_level=0.5,
            inference_timestep=0.1,
            num_policy_samples=128,
            task=env.task,
            num_samples=1,
            noise_level=0.3,
            plan_horizon=1.0,
            num_knots=policy.model.horizon,
        )

        mj_model = env.task.mj_model
        mj_data = mujoco.MjData(mj_model)
        run_sampling(ctrl, mj_model, mj_data, frequency=50, fixed_camera_id=0)

    else:
        parser.print_help()
