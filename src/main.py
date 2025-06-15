"""Main entrypoint for Offline Quantum Reinforcement Learning PoC."""

# import gymnasium as gym

from train import run_train

# from utils import initialize_rng


def main():
    """Main function to initialize and start training."""
    env_name = "CartPole-v1"
    num_epochs = 3
    seed = 0
    # rng = initialize_rng(seed)

    # Start training
    run_train(env_name, num_epochs, seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    # Call your training function here
    run_train(env_name=args.env_name, num_epochs=args.num_epochs, seed=args.seed)

    # remove the following comment to run manually (not using SLURM)
    # main()
