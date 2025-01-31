import random
from collections import deque

import torch


class Experience:
    """Class to store a single transition."""

    def __init__(self, obs, action, reward, next_obs, terminated):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.action = torch.tensor(action, dtype=torch.long)
        self.reward = torch.tensor(reward, dtype=torch.float32)
        self.next_obs = torch.tensor(next_obs, dtype=torch.float32)
        self.terminated = torch.tensor(terminated, dtype=torch.float32)


class ReplayMemory:
    """Replay memory for storing transitions."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
