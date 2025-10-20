import argparse

import mujoco
from flax import nnx
from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive as run_sampling

from gpc.architectures import DenoisingCNN
from gpc.envs import PushTEnv
from gpc.policy import Policy
from gpc.sampling import BootstrappedPredictiveSampling
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Push a T-shaped block on a table"
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
    env = PushTEnv(episode_length=400)
    save_file = "/tmp/pusht_policy.pkl"

    if args.task == "train":
        # Train the policy and save it to a file
        plan_horizon = 1.5
        num_knots = 15
        ctrl = PredictiveSampling(
            env.task,
            num_samples=128,
            noise_level=0.4,
            plan_horizon=plan_horizon,
            num_knots=num_knots,
        )
        net = DenoisingCNN(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=num_knots,
            feature_dims=[32, 32, 32],
            timestep_embedding_dim=8,
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            num_policy_samples=32,
            log_dir="/tmp/gpc_pusht",
            num_iters=20,
            num_envs=128,
            num_epochs=10,
            checkpoint_every=5,
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif args.task == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        mj_data = mujoco.MjData(env.task.mj_model)
        mj_data.qpos[:] = [0.1, 0.1, 2.0, 0.0, 0.0]  # set the initial state
        test_interactive(env, policy, mj_data)

    elif args.task == "sample":
        # Use the policy to bootstrap sampling-based MPC
        policy = Policy.load(save_file)
        ctrl = BootstrappedPredictiveSampling(
            policy,
            env.get_obs,
            warm_start_level=0.5,
            num_policy_samples=32,
            task=env.task,
            num_samples=1,
            noise_level=0.1,
            plan_horizon=1.5,
            num_knots=policy.model.horizon,
        )
        mj_model = env.task.mj_model
        mj_data = mujoco.MjData(mj_model)
        run_sampling(ctrl, mj_model, mj_data, frequency=50)

    else:
        parser.print_help()
