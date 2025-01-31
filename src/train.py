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
    reward_history = []

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} started...")  # Start of the epoch
        total_reward = 0
        epoch_reward = 0

        # Total number of steps in the dataset
        num_steps = dataset.size  # Assuming OfflineDataset has a `.size` attribute
        log_interval = max(1, num_steps // 20)  # Log every 5% of progress

        total_samples = dataset.size
        for idx, (state, action, reward, next_state, terminated) in enumerate(dataset):
            # Progress log every 5% (idx = step number)
            if idx % (total_samples // 20) == 0:  # 5% increments
                print(
                    f"Processing sample {idx}/{total_samples} ({(idx / total_samples) * 100:.1f}%)",
                )

            agent.memory.push(Experience(state, action, reward, next_state, terminated))
            agent.update()

        reward_history.append(epoch_reward)
        print(f"Epoch {epoch + 1}/{num_epochs} completed. Epoch Reward = {epoch_reward}")
