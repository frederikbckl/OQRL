"""Simplified training module."""

import gymnasium as gym

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

    dataset = "offline_cartpole_test_v5.hdf5"
    dataloader = HDF5Dataset(dataset)
    print("Note: dataset set up")

    # set up agent
    obs_space = env.observation_space  # get observation space
    act_space = env.action_space  # get action space
    agent = agent_fac.create(obs_space, act_space, rng)  # Create the agent instance
    print("Note: agent set up")

    # Run training
    for _ in range(num_epochs):
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
            print("Note: in training loop - dataloader")
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
            print("Note: in training loop - dataloader: finished agent.update")
            total_reward += rewards.sum().item()
            total_steps += len(states)
            print("Note: in training loop - dataloader: updated reward and steps")
            agent.on_step_end()
            print("Note: in training loop - dataloader: after on_step_end")

        agent.on_episode_end(
            total_steps,
            total_reward,
        )
        print("Note: in training loop: after on_episode_end")


# This updated version explicitly creates an agent from the factory before the training loop.
