"""Main entrypoint for Offline Quantum Reinforcement Learning PoC."""

from agent import DQNAgent
from train import run_train

# import gymnasium as gym


def main():
    """Main function to initialize and start training."""
    env_name = "CartPole-v1"
    num_epochs = 5
    seed = 0

    # Initialize the agent
    agent = DQNAgent(
        obs_dim=4,
        act_dim=2,
        learning_rate=0.001,
        gamma=0.99,
        replay_capacity=10000,
        batch_size=64,
        vqc_layers=2,
    )

    # Start training
    run_train(env_name, num_epochs, seed)


if __name__ == "__main__":
    main()
