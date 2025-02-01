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
        batch_idx = 0
        batch_size = 64
        last_logged_percentage = 0

        # Process dataset in batches instead of single samples
        for batch in dataset.get_batches(batch_size=64):
            print(f"Processing batch {batch_idx + 1}/{total_samples // batch_size}")  # Batch log
            states, actions, rewards, next_states, terminals = batch

            # Store in replay memory
            for i in range(len(states)):
                agent.memory.push(
                    Experience(states[i], actions[i], rewards[i], next_states[i], terminals[i]),
                )

            # Update agent
            agent.update()

            # Track progress
            # last_logged_percentage = 0

            # Log progress
            batch_idx += 1
            processed_samples = batch_idx * batch_size  # Total processed samples
            current_percentage = int((processed_samples / total_samples) * 100)

            # Log progress when crossing new 5% threshold
            if current_percentage >= last_logged_percentage + 5:
                print(
                    f"Processed {processed_samples}/{total_samples} samples ({current_percentage:.1f}%)",
                )
                last_logged_percentage = current_percentage

        reward_history.append(epoch_reward)
        print(f"Epoch {epoch + 1} completed. Total Reward = {epoch_reward}")

        # OLD CODE BELOW

        # Total number of steps in the dataset
        # num_steps = dataset.size  # Assuming OfflineDataset has a `.size` attribute
        # log_interval = max(1, num_steps // 20)  # Log every 5% of progress

        # total_samples = dataset.size

        # NEW PART OF OLD CODE START
        # Implement batching for the dataset to reduce overhead
        # for batch_idx, batch in enumerate(
        #     dataset.get_batches(batch_size=64)
        # ):  # Assuming `get_batches` is implemented
        #     states, actions, rewards, next_states, terminals = batch
        #     agent.memory.push_batch(states, actions, rewards, next_states, terminals)  # A new batch method
        #     agent.update()

        #     # Progress log every 5%
        #     if batch_idx % (dataset.size // (batch_size * 20)) == 0:  # 5% increments
        #         print(f"Processed {batch_idx * batch_size}/{dataset.size} samples")
        # # end of batching
        # NEW PART OF OLD CODE END

        # for idx, (state, action, reward, next_state, terminated) in enumerate(dataset):
        #     # Progress log every 5% (idx = step number)
        #     if idx % (total_samples // 20) == 0:  # 5% increments
        #         print(
        #             f"Processing sample {idx}/{total_samples} ({(idx / total_samples) * 100:.1f}%)",
        #         )

        #     agent.memory.push(Experience(state, action, reward, next_state, terminated))
        #     agent.update()

        # reward_history.append(epoch_reward)
        # print(f"Epoch {epoch + 1}/{num_epochs} completed. Epoch Reward = {epoch_reward}")
