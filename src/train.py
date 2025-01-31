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

        for step, (state, action, reward, next_state, terminated) in enumerate(dataset):
            agent.memory.push(Experience(state, action, reward, next_state, terminated))
            agent.update()
            epoch_reward += reward

            # Log progress every 5% or at the last step
            if (step + 1) % log_interval == 0 or (step + 1) == num_steps:
                percentage_done = ((step + 1) / num_steps) * 100
                print(f"Epoch {epoch + 1}/{num_epochs}: {percentage_done:.1f}% done")

        reward_history.append(epoch_reward)
        print(f"Epoch {epoch + 1}/{num_epochs} completed. Epoch Reward = {epoch_reward}")

    # Training loop (old)
    # for epoch in range(num_epochs):
    #     print(f"Epoch {epoch + 1}/{num_epochs} started...")  # Start of the epoch
    #     state = env.reset()[0]
    #     total_reward = 0
    #     epoch_reward = 0

    #     for state, action, reward, next_state, terminated in dataset:
    #         agent.memory.push(Experience(state, action, reward, next_state, terminated))
    #         agent.update()
    #         epoch_reward += reward
    #     reward_history.append(epoch_reward)
    #     # agent.update_target()
    #     print(f"Epoch {epoch + 1}: Epoch Reward = {epoch_reward}")
    #     print(f"Epoch {epoch + 1}: Total Reward = {total_reward}")
    # # end of Training loop (old)

    # Training loop v2
    # for epoch in range(num_epochs):
    #     total_reward = 0.0
    #     for obs, action, reward, next_obs, terminal in dataset:
    #         agent.update(obs, action, reward, next_obs, terminal)
    #         total_reward += reward
    #     reward_history.append(total_reward)
    #     print(f"Epoch {epoch + 1}/{num_epochs}, Total Reward: {total_reward}")
    # print("Training completed.")
