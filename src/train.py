import gymnasium as gym
import torch

from agent import DQNAgent
from config import BATCH_SIZE, MAX_INTERACTIONS, POLICY_UPDATE_FREQUENCY, TARGET_UPDATE_FREQUENCY
from dataset import OfflineDataset
from optim import GAOptimizer  # Import GAOptimizer
from utils import Experience, device, initialize_rng

"""Training module for simplified Offline QRL."""


def run_train(env_name, num_epochs, seed):
    """Run training for the given environment."""
    # Initialize environment
    env = gym.make(env_name, render_mode=None)  # Disable rendering for consistency
    env.reset(seed=seed)  # Seed the environment at creation

    rng = initialize_rng(seed)

    # Initialize the agent
    agent = DQNAgent(
        obs_dim=4,
        act_dim=2,
        learning_rate=0.001,
        gamma=0.99,
        replay_capacity=10000,
        batch_size=BATCH_SIZE,
        policy_update_frequency=POLICY_UPDATE_FREQUENCY,
        target_update_frequency=TARGET_UPDATE_FREQUENCY,
        vqc_layers=2,
        rng=rng,
    )

    # Replace the optimizer with respective Metaheuristic Optimizer
    # OLD
    # agent.optimizer = GAOptimizer(agent.policy_net, rng=rng)

    # NEW
    agent.optimizer = GAOptimizer(
        model=agent.policy_net,
        rng=rng,
    )

    # Load dataset
    dataset_path = "offline_cartpole_v2.hdf5"
    dataset = OfflineDataset(dataset_path, rng=rng)
    total_samples = dataset.size
    reward_history = []
    # subset_fraction = 0.05  # Fraction of the dataset to use for training
    # subset_size = int(total_samples * subset_fraction)
    batch_size = BATCH_SIZE
    max_interactions = MAX_INTERACTIONS

    print(f"Total samples: {total_samples}")
    # print(f"Subset size: {subset_size}")
    # print(f"Max batches this epoch: {subset_size // batch_size}")

    # Training loop
    for epoch in range(num_epochs):
        print("\n--------------------------------")
        print(f"Epoch {epoch + 1}/{num_epochs} started...")  # Start of the epoch
        print("--------------------------------\n")

        # Reset update_counter
        agent.update_counter = 0

        # sync target network with policy network
        print(
            f"\n[TARGET] Updating target network at batch {agent.update_counter}",
        )
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # print(f"Max batches this epoch: {subset_size // batch_size}")

        # Sample a fresh subset from the full dataset
        # subset = dataset.sample(subset_size)
        # print(f"Sampled {len(subset)} experiences for this epoch.")

        # print(f"Dataset size used for training: {subset_size}")
        epoch_reward = 0
        batch_idx = 1
        processed_samples = 0

        while agent.optimizer.interaction_count < max_interactions:
            batch = dataset.sample(batch_size)
            if len(batch) == 0:
                continue

            # Process batch
            states, actions, rewards, next_states, terminals = zip(*batch)

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
            states = torch.stack(states).to(device)
            actions = torch.stack(actions).to(device)
            rewards = torch.stack(rewards).to(device)
            next_states = torch.stack(next_states).to(device)
            terminals = torch.stack(terminals).to(device)

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

            # Update agent
            agent.update()

            # Accumulate rewards (for the epoch)
            batch_reward = sum(rewards)
            epoch_reward += batch_reward  # .item()

            # Accumulate total samples processed
            processed_samples += len(states)

            if hasattr(agent.optimizer, "interaction_count"):
                print(
                    f"[Interaction Log] Total counted dataset interactions so far: {agent.optimizer.interaction_count}",
                )

            batch_idx += 1

        reward_history.append(epoch_reward)
        print(f"\nEpoch {epoch + 1} completed. Total Reward = {epoch_reward:.2f}")
        print(f"Dataset interactions in Epoch {epoch+1}: {agent.optimizer.interaction_count}\n")

        # Testing the agent
        print(f"Starting evaluation for Epoch {epoch + 1}...")
        eval_rewards = []
        num_episodes = 25  # Number of episodes for evaluation
        for episode in range(num_episodes):
            state = env.reset(seed=seed + episode)[0]  # new: seeded env.reset
            done = False
            episode_reward = 0
            while not done:
                action = agent.act(state, epsilon=0.0)  # Use the policy to select an action
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
