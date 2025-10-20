import argparse

import mujoco
from flax import nnx
from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive as run_sampling

from gpc.architectures import DenoisingCNN
from gpc.envs import CraneEnv
from gpc.policy import Policy
from gpc.sampling import BootstrappedPredictiveSampling
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Swing the payload of a luffing crane to a target position."
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
    env = CraneEnv(episode_length=500)
    save_file = "/tmp/crane_policy.pkl"

    if args.task == "train":
        # Train the policy and save it to a file
        plan_horizon = 1.5
        num_knots = 15
        ctrl = PredictiveSampling(
            env.task,
            num_samples=4,
            noise_level=0.05,
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
            num_policy_samples=2,
            log_dir="/tmp/gpc_crane",
            num_iters=10,
            num_envs=512,
            num_videos=4,
            batch_size=1024,
            num_epochs=20,
            checkpoint_every=5,
            strategy="policy",
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif args.task == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        test_interactive(env, policy)

    elif args.task == "sample":
        # Use the policy to bootstrap sampling-based MPC
        policy = Policy.load(save_file)
        ctrl = BootstrappedPredictiveSampling(
            policy,
            env.get_obs,
            inference_timestep=0.1,
            num_policy_samples=4,  # samples from the flow-matching policy
            task=env.task,
            num_samples=4,  # samples from a gaussian around the previous mean
            noise_level=0.05,
            plan_horizon=1.5,
            num_knots=policy.model.horizon,
        )
        mj_model = env.task.mj_model
        mj_data = mujoco.MjData(mj_model)
        run_sampling(ctrl, mj_model, mj_data, frequency=30)

    else:
        parser.print_help()
