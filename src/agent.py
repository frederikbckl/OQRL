import numpy as np
import pennylane as qml
import torch
from torch import nn, optim

from utils import ReplayMemory

# from utils import Experience


class VQC(nn.Module):
    """Variational Quantum Circuit implemented using PennyLane."""

    def __init__(self, input_dim, output_dim, n_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.params = nn.Parameter(torch.rand(n_layers * input_dim * 3, requires_grad=True))
        self.device = torch.device("cpu")

        self.dev = qml.device("default.qubit", wires=input_dim)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, inputs, weights):
        """Define the quantum circuit."""
        # Ensure the weights tensor has the correct shape
        # print("Initial weights shape:", weights.shape)
        # print("Expected shape:", (self.n_layers, self.input_dim, 3))
        reshaped_weights = weights.reshape(
            self.n_layers,
            self.input_dim,
            3,
        )  # Adjust last dimension to 3
        qml.AngleEmbedding(inputs, wires=range(self.input_dim))
        qml.StronglyEntanglingLayers(reshaped_weights, wires=range(self.input_dim))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]

    def forward(self, x):
        # print(f"RunningVQC forward pass for batch of size {len(x)}")
        outputs = []
        for sample in x:
            # Ensure the output is a PyTorch tensor
            result = self.qnode(sample, self.params)
            result_tensor = (
                torch.tensor(result) if not isinstance(result, torch.Tensor) else result
            )
            outputs.append(result_tensor)
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

        # Unpack Experience objects
        states = torch.stack([exp.obs for exp in batch])
        actions = torch.stack([exp.action for exp in batch])
        rewards = torch.stack([exp.reward for exp in batch])
        next_states = torch.stack([exp.next_obs for exp in batch])
        terminals = torch.stack([exp.terminated for exp in batch])

        # Compute Q-values (no gradients required)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0]
        targets = rewards + (1 - terminals) * self.gamma * next_q_values

        # Simulated Annealing optimization
        def loss_fn():
            return torch.mean((q_values - targets) ** 2).item()

        # perform simlated annealin step
        self.optimizer.step(loss_fn)

        # Update (old)
        # loss = self.loss_fn(q_values, targets.detach())
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

    def update_target(self):
        """Update the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
