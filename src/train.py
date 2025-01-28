"""Training module for simplified Offline QRL."""

import gymnasium as gym
import numpy as np
import pennylane as qml
import torch
from torch import nn, optim

from dataset import HDF5Dataset
from utils import ReplayMemory


def run_train(agent_fac, env_name, num_epochs, dataset_path):
    """Run training for the given environment."""
    # Initialize environment
    env = gym.make(env_name)
    obs_space = env.observation_space
    act_space = env.action_space

    # Initialize agent
    agent = agent_fac.create(obs_space, act_space)

    # Load dataset
    dataset = HDF5Dataset(dataset_path)
    reward_history = []

    # Training loop
    for epoch in range(num_epochs):
        total_reward = 0.0
        for obs, action, reward, next_obs, terminal in dataset:
            agent.update(obs, action, reward, next_obs, terminal)
            total_reward += reward
        reward_history.append(total_reward)
        print(f"Epoch {epoch + 1}/{num_epochs}, Total Reward: {total_reward}")

    print("Training completed.")


class VQC(nn.Module):
    """Variational Quantum Circuit implemented using PennyLane."""

    def __init__(self, input_dim, output_dim, n_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.params = nn.Parameter(torch.rand(n_layers * input_dim * 2, requires_grad=True))
        self.device = torch.device("cpu")

        self.dev = qml.device("default.qubit", wires=input_dim)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(self.input_dim))
        qml.StronglyEntanglingLayers(
            weights.reshape(self.n_layers, self.input_dim, 2),
            wires=range(self.input_dim),
        )
        return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]

    def forward(self, x):
        outputs = []
        for sample in x:
            outputs.append(self.qnode(sample, self.params))
        return torch.stack(outputs)


class DQNAgent:
    """Deep Q-Learning Agent with VQC."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        learning_rate,
        gamma,
        replay_capacity,
        batch_size,
        vqc_layers=2,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.batch_size = batch_size

        # Initialize VQC policy network
        self.policy_net = VQC(obs_dim, act_dim, n_layers=vqc_layers)
        self.target_net = VQC(obs_dim, act_dim, n_layers=vqc_layers)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay memory
        self.memory = ReplayMemory(replay_capacity)

    def act(self, state, epsilon=0.1):
        """Select an action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.act_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def update(self):
        """Update the policy network using a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, terminals = map(torch.stack, zip(*batch))

        # Compute Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0]
        targets = rewards + (1 - terminals) * self.gamma * next_q_values

        # Update
        loss = self.loss_fn(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        """Update the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
