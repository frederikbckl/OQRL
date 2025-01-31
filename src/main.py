"""Main entrypoint for Offline Quantum Reinforcement Learning PoC."""

from train import run_train

# import gymnasium as gym


def main():
    """Main function to initialize and start training."""
    env_name = "CartPole-v1"
    num_epochs = 5
    seed = 0

    # Start training
    run_train(env_name, num_epochs, seed)


if __name__ == "__main__":
    main()
