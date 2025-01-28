"""Simplified training module."""

import gymnasium as gym
import matplotlib.pyplot as plt

from agent import AgentFactory

# from torch.utils.data import DataLoader
from dataset import HDF5Dataset

# from dataset import RLDataset
from experience import Experience
from rng import initialize_rng
from utils import ReplayMemory


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

    # set up replay memory
    memory_capacity = 10000  # assuming we have a capacity defined or pass it as a parameter
    replay_memory = ReplayMemory(rng, memory_capacity)

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
            print("Note: in dataloader loop")
            print(
                "States type:",
                type(states),
                "Length:",
                len(states) if hasattr(states, "__len__") else "Not iterable",
            )
            print("Actions type:", type(actions))
            print("Rewards type:", type(rewards))
            print("Terminals type:", type(terminals))

            # Add the experience to the replay memory
            for i in range(len(states)):  # Assuming batch size is iterable
                # Extract reward directly without indexing
                reward = rewards  # No [i] because it's a scalar tensor
                # if isinstance(reward, torch.Tensor):
                #     reward = reward.item()  # Convert tensor to scalar (handles 0-dim tensor)
                terminated = terminals  # No [i] because it's a scalar tensor

                # Push experience to replay memory
                replay_memory.push(
                    Experience(
                        obs=states[i],
                        action=actions[i],
                        reward=reward,  # Use the processed reward
                        next_obs=next_states[i],
                        terminated=terminated,  # Test: No [i] because it's a scalar tensor
                        info={},  # Add empty dictionary for the info argument
                    ),
                )
            # Perform updates, etc. (already exists in your code)
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
