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
    rng = initialize_rng(seed)

    # set up environment and dataset
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Failed to create environment {env_name}: {e}")
        raise

    dataset = "offline_cartpole_test_v5.hdf5"
    dataloader = HDF5Dataset(dataset)

    # set up agent
    obs_space = env.observation_space  # get observation space
    act_space = env.action_space  # get action space
    agent = agent_fac.create(obs_space, act_space, seed)  # Create the agent instance

    # Run training
    for _ in range(num_epochs):
        total_reward = 0.0
        total_steps = 0
        for (
            states,
            actions,
            rewards,
            next_states,
            terminals,
        ) in dataloader:  # left out timeouts due to testing
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
        agent.on_episode_end(
            total_steps,
            total_reward,
        )
