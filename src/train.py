"""Training module for simplified Offline QRL."""

import gymnasium as gym

from agent import DQNAgent
from dataset import OfflineDataset
from utils import Experience

"""Training module for simplified Offline QRL."""


def run_train(env_name, num_epochs, seed):
    """Run training for the given environment."""
    # Initialize environment
    env = gym.make(env_name)

    # obs_space = env.observation_space
    # act_space = env.action_space

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

    # Load dataset
    dataset_path = "offline_cartpole_v2.hdf5"
    dataset = OfflineDataset(dataset_path)
    total_samples = dataset.size
    reward_history = []

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} started...")  # Start of the epoch
        # total_reward = 0
        epoch_reward = 0
        batch_idx = 1
        batch_size = 64
        last_logged_percentage = 0

        # Process dataset in batches instead of single samples
        for batch in dataset.get_batches(batch_size=64):
            # print(f"Processing batch {batch_idx}/{total_samples // batch_size}")  # Batch log
            states, actions, rewards, next_states, terminals = batch

            # Store in replay memory
            for i in range(len(states)):
                agent.memory.push(
                    Experience(states[i], actions[i], rewards[i], next_states[i], terminals[i]),
                )

            # Update agent
            agent.update()

            # Accumulate rewards (for the epoch)
            batch_reward = rewards.sum()
            epoch_reward += batch_reward  # .item()

            # print(f"Batch {batch_idx} completed. Batch Reward = {batch_reward:.2f}")

            # Log progress every 5% of the dataset
            processed_samples = batch_idx * len(states)  # Total processed samples
            # replaced batch_size with len(states) for accurate count (last batch may be smaller)
            if processed_samples % (total_samples // 20) == 0:  # Every 5% of the dataset
                print(
                    f"Processed {processed_samples}/{total_samples} samples "
                    f"({(processed_samples / total_samples) * 100:.1f}%) - "
                    f"Current Epoch Reward: {epoch_reward:.2f}",
                )

            # Log progress when crossing new 5% threshold
            # if current_percentage >= last_logged_percentage + 5:
            #     print(
            #         f"Processed {processed_samples}/{total_samples} samples ({current_percentage:.1f}%)",
            #     )
            #     last_logged_percentage = current_percentage

            batch_idx += 1

        reward_history.append(epoch_reward)
        print(f"Epoch {epoch + 1} completed. Total Reward = {epoch_reward:.2f}")
