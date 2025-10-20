import argparse

import mujoco
from flax import nnx
from hydrax.algs import MPPI
from hydrax.simulation.deterministic import run_interactive as run_sampling

from gpc.architectures import DenoisingCNN
from gpc.envs import HumanoidEnv
from gpc.policy import Policy
from gpc.sampling import BootstrappedPredictiveSampling
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Humanoid standup from arbitrary initial positions"
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
    env = HumanoidEnv(episode_length=400)
    save_file = "/tmp/humanoid_policy.pkl"

    if args.task == "train":
        # Train the policy and save it to a file
        plan_horizon = 1.0
        num_knots = 10
        ctrl = MPPI(
            env.task,
            num_samples=32,
            noise_level=1.0,
            temperature=0.1,
            num_randomizations=2,
            plan_horizon=plan_horizon,
            num_knots=num_knots,
        )
        net = DenoisingCNN(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=num_knots,
            feature_dims=(128,) * 3,
            timestep_embedding_dim=64,
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            num_policy_samples=32,
            log_dir="/tmp/gpc_humanoid",
            num_epochs=10,
            num_iters=50,
            num_envs=128,
            num_videos=2,
            checkpoint_every=1,
            strategy="best",
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
            num_policy_samples=128,
            warm_start_level=0.9,
            task=env.task,
            num_samples=128,
            noise_level=0.5,
            num_randomizations=2,
            plan_horizon=1.0,
            num_knots=policy.model.horizon,
        )

        mj_model = env.task.mj_model
        mj_model.opt.timestep = 0.01

        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[3:7] = [-0.7, 0.0, 0.7, 0.0]

        run_sampling(ctrl, mj_model, mj_data, frequency=50, show_traces=False)

    else:
        parser.print_help()
