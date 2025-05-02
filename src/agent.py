# import time

import numpy as np
import pennylane as qml
import torch
from torch import nn

from config import device
from optim import GAOptimizer

# from optim import SimulatedAnnealing
from utils import ReplayMemory

# from utils import Experience


class VQC(nn.Module):
    """Variational Quantum Circuit implemented using PennyLane."""

    def __init__(self, input_dim, output_dim, n_layers=2):
        """Reduced n_layers to 2 for faster training."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        # self.params = nn.Parameter(
        #     torch.rand(n_layers * input_dim * 3, requires_grad=True),
        # )  # forAdam
        self.params = nn.Parameter(torch.rand(n_layers * input_dim * 3), requires_grad=False)
        # self.device = torch.device(
        # "cuda" if torch.cuda.is_available() else "cpu",
        # )  # ✔️ Move device assignment

        self.device = device
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
        qml.AngleEmbedding(inputs, wires=range(self.input_dim), rotation="Y")
        qml.StronglyEntanglingLayers(reshaped_weights, wires=range(self.input_dim))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]

    def forward(self, x):
        # print(f"RunningVQC forward pass for batch of size {len(x)}")
        outputs = []
        for sample in x:
            # start_time = time.time()
            # Ensure the output is a PyTorch tensor
            result = self.qnode(sample, self.params)
            result_tensor = (
                torch.tensor(result) if not isinstance(result, torch.Tensor) else result
            )
            outputs.append(result_tensor)
            # print(f"VQC forward pass took {time.time() - start_time:.2f} seconds")

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
        self.replay_capacity = replay_capacity
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.update_counter = 0  # track how many times update() was called
        self.update_frequency = 64  # optimize every X updates

        # Initialize VQC policy network
        self.policy_net = VQC(obs_dim, act_dim, n_layers=vqc_layers).to(self.device)
        self.target_net = VQC(obs_dim, act_dim, n_layers=vqc_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Disable gradients for policy_net and target_net
        for param in self.policy_net.parameters():
            param.requires_grad = False
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = GAOptimizer(model=self.policy_net)
        self.loss_fn = nn.MSELoss()

        # Replay memory
        self.memory = ReplayMemory(replay_capacity)

    def act(self, state, epsilon=0.1):
        """Select an action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.act_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor.to(self.device))
        return q_values.argmax().item()

    def update(self):
        """Update the policy network using a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        self.update_counter += 1

        # Sample a batch from memory
        batch = self.memory.sample(self.batch_size)

        # Unpack batch into individual tensors
        states = torch.stack([exp.obs for exp in batch]).to(self.device)
        actions = torch.stack([exp.action for exp in batch]).to(self.device)
        rewards = torch.stack([exp.reward for exp in batch]).to(self.device)
        next_states = torch.stack([exp.next_obs for exp in batch]).to(self.device)
        terminals = torch.stack([exp.terminated for exp in batch]).to(self.device)

        # ensure actions and states are both on the same device before computing q_values
        print(f"States device before: {states.device}")
        states = states.to(device)
        print(f"States device after: {states.device}")

        print(f"Actions device before: {actions.device}")
        actions = actions.to(device)
        print(f"Actions device after: {actions.device}")

        # Compute Q-values for current states and actions (after states and actions are on same device)
        q_values = self.policy_net(states.to(device)).gather(
            1,
            actions.unsqueeze(1).to(device),
        )

        # Compute Q-values for current states and actions
        # q_values = (
        #     self.policy_net(states.to(self.device)).gather(1, actions.unsqueeze(1)).to(self.device)
        # )  # added device, might have to delete the last .squeeze()

        # Compute next Q-values only once (no gradients required)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]

        # Compute targets
        targets = rewards + (1 - terminals) * self.gamma * next_q_values

        # Define the loss function for metaheuristic optimization
        loss_fn = lambda: torch.nn.functional.mse_loss(q_values, targets).item()

        # Run GA optimization only every N updates
        if self.update_counter % self.update_frequency == 0:
            print(f"[UPDATE] Performing GA optimization at batch {self.update_counter}")
            self.optimizer.optimize(loss_fn, batch)
