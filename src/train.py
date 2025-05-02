import math

import gymnasium as gym
import torch

from agent import DQNAgent
from config import device
from dataset import OfflineDataset
from optim import GAOptimizer  # Import GAOptimizer
from utils import Experience

"""Training module for simplified Offline QRL."""


def run_train(env_name, num_epochs, seed):
    """Run training for the given environment."""
    # Initialize environment
    env = gym.make(env_name)

    # Set device (either cuda if available, else cpu)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    subset_fraction = 0.05  # Fraction of the dataset to use for training
    subset_size = int(total_samples * subset_fraction)
    batch_size = 64
    subset = dataset.sample(subset_size)
    max_batches = len(subset)  # since subset already contains the batches
    max_batches = subset_size // batch_size + int(subset_size % batch_size != 0)

    print(f"Total samples: {total_samples}")
    print(f"Subset size: {subset_size}")
    print(f"Max batches this epoch: {subset_size // batch_size}")

    # Training loop
    for epoch in range(num_epochs):
        print("\n--------------------------------")
        print(f"Epoch {epoch + 1}/{num_epochs} started...")  # Start of the epoch
        print("--------------------------------\n")

        # Reset update_counter
        agent.update_counter = 0

        print(f"Max batches this epoch: {subset_size // batch_size}")

        # Sample a fresh subset from the full dataset
        subset = dataset.sample(subset_size)
        print(f"Sampled {len(subset)} experiences for this epoch.")

        print(f"Dataset size used for training: {subset_size}")
        epoch_reward = 0
        batch_idx = 1
        last_logged_percentage = 0
        processed_samples = 0

        # Process dataset in batches instead of single samples

        for step, batch_start in enumerate(range(0, len(subset), batch_size), start=1):
            batch = subset[batch_start : batch_start + batch_size]
            if len(batch) == 0:
                continue

            # Process batch
            states, actions, rewards, next_states, terminals = zip(*batch)

            # NEW
            # Convert each element in the batch to a tensor
            states = [
                torch.tensor(state, dtype=torch.float32).to(device)
                if not isinstance(state, torch.Tensor)
                else state.to(device)
                for state in states
            ]

            actions = [
                torch.tensor(action, dtype=torch.int64).to(device)
                if not isinstance(action, torch.Tensor)
                else action.to(device)
                for action in actions
            ]

            rewards = [
                torch.tensor(reward, dtype=torch.float32).to(device)
                if not isinstance(reward, torch.Tensor)
                else reward.to(device)
                for reward in rewards
            ]

            next_states = [
                torch.tensor(next_state, dtype=torch.float32).to(device)
                if not isinstance(next_state, torch.Tensor)
                else next_state.to(device)
                for next_state in next_states
            ]

            terminals = [
                torch.tensor(terminal, dtype=torch.float32).to(device)
                if not isinstance(terminal, torch.Tensor)
                else terminal.to(device)
                for terminal in terminals
            ]

            # Move each tensor in the batch to the appropriate device
            # print(f"States device before: {states.device}")
            states = torch.stack(states).to(device)
            print(f"States device after moving to device: {states.device}")
            # print(f"Actions device before: {actions.device}")
            actions = torch.stack(actions).to(device)
            print(f"Actions device after moving to device: {actions.device}")
            rewards = torch.stack(rewards).to(device)
            print(f"Rewards device after moving to device: {rewards.device}")
            next_states = torch.stack(next_states).to(device)
            print(f"Next states device after moving to device: {next_states.device}")
            terminals = torch.stack(terminals).to(device)
            print(f"Terminals device after moving to device: {terminals.device}")

            # Store in replay memory and move tensors to the appropriate device
            for j in range(len(states)):
                agent.memory.push(
                    Experience(
                        states[j].to(device),
                        actions[j].to(device),
                        rewards[j].to(device),
                        next_states[j].to(device),
                        terminals[j].to(device),
                    ),
                )

            # OLD
            # store in replay memory
            # for j in range(len(states)):
            #     agent.memory.push(
            #         Experience(states[j], actions[j], rewards[j], next_states[j], terminals[j]),
            #     )

            # Update agent using GAOptimizer
            def loss_fn(q, target):
                return torch.nn.functional.mse_loss(q, target)

            # (f"Processing batch {batch_idx}/{max_batches} with {len(batch[0])} samples...",)

            # NEW: might be dead code but worth a try. Delete if not needed
            # Ensure that actions are moved to the correct device before calling agent.update
            actions = torch.tensor(actions).to(device)

            # Update agent
            agent.update()

            # Accumulate rewards (for the epoch)
            batch_reward = sum(rewards)
            epoch_reward += batch_reward  # .item()

            # Accumulate total samples processed
            processed_samples += len(states)
            current_percentage = (processed_samples / total_samples) * 100

            log_interval = 10  # every 10% of the subset
            subset_log_step = math.ceil(subset_size * log_interval / 100)

            if processed_samples % subset_log_step < batch_size:
                print(
                    f"Processed {processed_samples}/{subset_size} samples "
                    f"({(processed_samples / subset_size) * 100:.1f}%) of current subset",
                )

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
