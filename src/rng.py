"""RNG functions."""

import random
from os import environ

import numpy as np
import torch
from numpy.random import Generator


def initialize_rng(seed_value: int) -> Generator:
    """Seeds Random, Numpy, PyTorch and returns a Numpy Generator."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.manual_seed_all(seed_value)
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed_value)
