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
    return np.random.default_rng(seed_value)


class Experience:
    """Class to store a single transition."""

    # NEW: add device
    # Set device (either cuda if available, else cpu)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NEW: try to get rid of warnings
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

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
