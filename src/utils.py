import random
from collections import deque
from os import environ

import numpy as np
import torch
from numpy.random import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_rng(seed_value: int) -> Generator:
    """Seeds Random, Numpy, PyTorch and returns a Numpy Generator"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.manual_seed_all(seed_value)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed_value)


class Experience:
    """Class to store a single transition."""

    # to get rid of warnings
    def __init__(self, obs, action, reward, next_obs, terminated):
        self.obs = obs.detach().clone().to(device)
        self.action = action.detach().clone().to(device)
        self.reward = reward.detach().clone().to(device)
        self.next_obs = next_obs.detach().clone().to(device)
        self.terminated = terminated.detach().clone().to(device)

    # def __init__(self, obs, action, reward, next_obs, terminated):
    #     self.obs = torch.tensor(obs, dtype=torch.float32)
    #     self.action = torch.tensor(action, dtype=torch.long)
    #     self.reward = torch.tensor(reward, dtype=torch.float32)
    #     self.next_obs = torch.tensor(next_obs, dtype=torch.float32)
    #     self.terminated = torch.tensor(terminated, dtype=torch.float32)


class ReplayMemory:
    """Replay memory for storing transitions."""

    def __init__(self, capacity, rng):
        self.memory = deque(maxlen=capacity)
        self.rng = rng

    def push(self, experience: Experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        # seeded sample function
        indices = self.rng.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]

    def __len__(self):
        return len(self.memory)
