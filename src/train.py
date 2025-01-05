"""Simplified training module."""

import gymnasium as gym
from torch.utils.data import DataLoader

from config import BATCH_SIZE


def run_train(agent_fac, env_name, num_epochs, seed):
    """Run a very basic training loop."""
    # TODO: initialize rng / seed

    # set up environment and dataset
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Failed to create environment {env_name}: {e}")
        raise

    dataset = "offline_cartpole_test_v5.hdf5"
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # set up agent
    obs_space = env.observation_space  # get observation space
    act_space = env.action_space  # get action space
    agent = agent_fac.create(obs_space, act_space, seed)  # Create the agent instance

    obs = env.reset()
    action = agent.policy(obs)  # Assuming your agent's policy method is correctly implemented
    print(env.step(action))  # This print statement is to debug what env.step returns


# This updated version explicitly creates an agent from the factory before the training loop.
