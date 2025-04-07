import gymnasium as gym
import torch

from agent import DQNAgent
from dataset import OfflineDataset
from optim import GAOptimizer  # Import GAOptimizer
from utils import Experience

"""Training module for simplified Offline QRL."""


def run_train(env_name, num_epochs, seed):
    """Run training for the given environment."""
    # Initialize environment
    env = gym.make(env_name)

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

    # Replace the optimizer with GAOptimizer
    agent.optimizer = GAOptimizer(agent.policy_net)

    # Load dataset
    dataset_path = "offline_cartpole_v2.hdf5"
    dataset = OfflineDataset(dataset_path)
    total_samples = dataset.size
    reward_history = []
    subset_fraction = 0.3  # Fraction of the dataset to use for training
    subset_size = int(total_samples * subset_fraction)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} started...")  # Start of the epoch
        epoch_reward = 0
        batch_idx = 1
        # batch_size = 64
        last_logged_percentage = 0

        # Process dataset in batches instead of single samples
        for batch in dataset.get_batches(batch_size=64):
            if batch_idx > subset_size // 64:  # Limit the number of batches
                break

            # Process batch
            states, actions, rewards, next_states, terminals = batch

            # Store in replay memory
            for i in range(len(states)):
                agent.memory.push(
                    Experience(states[i], actions[i], rewards[i], next_states[i], terminals[i]),
                )

            # Update agent using GAOptimizer
            def loss_fn(q, target):
                return torch.nn.functional.mse_loss(q, target)

            agent.optimizer.optimize(loss_fn, batch)

            # Update agent
            agent.update()

            # Accumulate rewards (for the epoch)
            batch_reward = rewards.sum()
            epoch_reward += batch_reward  # .item()

            # Log progress every 5% of the dataset
            processed_samples = batch_idx * len(states)  # Total processed samples
            # replaced batch_size with len(states) for accurate count (last batch may be smaller)
            current_percentage = (processed_samples / total_samples) * 100

            if current_percentage >= last_logged_percentage + 5:
                print(
                    f"Processed {processed_samples}/{total_samples} samples "
                    f"({current_percentage:.1f}%)",
                )
                last_logged_percentage += 5

            batch_idx += 1

        reward_history.append(epoch_reward)
        print(f"Epoch {epoch + 1} completed. Total Reward = {epoch_reward:.2f}")

        # Testing the agent
        print(f"Starting evaluation for Epoch {epoch + 1}...")
        eval_rewards = []
        num_episodes = 25  # Number of episodes for evaluation
        for episode in range(num_episodes):
            state = env.reset()[0]
            done = False
            episode_reward = 0
            while not done:
                action = agent.act(state)  # Use the policy to select an action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += float(reward)
                state = next_state
            eval_rewards.append(episode_reward)
            print(
                f"Evaluation Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}",
            )

        # Logging evaluation statistics
        best_reward = max(eval_rewards)
        worst_reward = min(eval_rewards)
        avg_reward = sum(eval_rewards) / len(eval_rewards)
        print(
            f"Epoch {epoch + 1} Evaluation Results:\n"
            f"  Best Reward: {best_reward:.2f}\n"
            f"  Worst Reward: {worst_reward:.2f}\n"
            f"  Average Reward: {avg_reward:.2f}\n"
            f"  Total Episodes: {num_episodes}\n",
        )
