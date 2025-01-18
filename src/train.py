"""Simplified training module."""

import gymnasium as gym
import matplotlib.pyplot as plt

from agent import AgentFactory

# from torch.utils.data import DataLoader
from dataset import HDF5Dataset

# from dataset import RLDataset
from experience import Experience
from rng import initialize_rng


def run_train(
    agent_fac: AgentFactory,
    env_name: str,
    num_epochs: int,
    seed: int,
) -> None:
    """Run training for the given environment."""
    print("Note: run_train started")
    rng = initialize_rng(seed)
    print("Note: initialized rng")

    # set up environment and dataset
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Failed to create environment {env_name}: {e}")
        raise
    print("Note: created environment")

    dataset = "offline_cartpole_v2.hdf5"
    dataloader = HDF5Dataset(dataset)
    print("Note: dataset set up")

    # set up agent
    obs_space = env.observation_space  # get observation space
    act_space = env.action_space  # get action space
    agent = agent_fac.create(obs_space, act_space, rng)  # Create the agent instance
    print("Note: agent set up")

    reward_history = []

    # Run training
    for epoch in range(num_epochs):
        print("Note: in training loop")
        total_reward = 0.0
        total_steps = 0
        for (
            states,
            actions,
            rewards,
            next_states,
            terminals,
        ) in dataloader:  # left out timeouts due to testing
            # print("Agent weights before update:", agent.policy_net.state_dict())
            agent.update(
                Experience(
                    states,
                    actions,
                    rewards,
                    next_states,
                    terminals,
                    {},
                ),  # left out timeouts again
            )
            total_reward += rewards.sum().item()
            total_steps += len(states)
            agent.on_step_end()
            # print(f"Note: Training Loop - Step: {total_steps}")
            # print("Agent weights after update:", agent.policy_net.state_dict())

        reward_history.append(total_reward)  # Save total reward for this epoch
        print(f"Epoch {epoch + 1} completed with total reward: {total_reward}")
        agent.on_episode_end(
            total_steps,
            total_reward,
        )

    plt.plot(range(1, num_epochs + 1), reward_history)
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()


# This updated version explicitly creates an agent from the factory before the training loop.
