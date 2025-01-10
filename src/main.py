"""Main python file for offline quantum reinforcement learning PoC."""

# import numpy as np

from agent import AgentFactory
from dqn import DQN
from train import run_train


def main() -> None:
    """Initialize the Main entrypoint of the program."""
    print("Note: main started")
    env_name = "CartPole-v1"  # Setting environment to CartPole v1
    agent_fac = AgentFactory(DQN)  # Using DQN as the agent model
    num_epochs = 1000
    seed = 0
    print("Note: agent_fac, env_name, num_epochs, seed initialized")
    run_train(agent_fac, env_name, num_epochs, seed)  # Start the training process


if __name__ == "__main__":
    main()
