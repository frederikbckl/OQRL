import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        """Initialize the dataset by loading the HDF5 file.

        Args: file_path (str): Path to the HDF5 file.
        """
        # print("Hello __init__")
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as f:
            self.observations = np.array(f["observations"])
            self.actions = np.array(f["actions"])
            self.rewards = np.array(f["rewards"])
            # Debug prints
            print("Rewards type after loading from HDF5:", type(self.rewards))
            print("Rewards shape after loading from HDF5:", self.rewards.shape)
            print("First few rewards:", self.rewards[:5])  # Print a few sample values
            self.next_observations = np.array(f["next_observations"])
            self.terminals = np.array(f["terminals"])

    def __len__(self):
        """Return the size of the dataset."""
        # print("Hello __len__")
        return len(self.observations)

    def __getitem__(self, idx):
        """Return a single sample of data."""
        # print("Hello __getitem__")
        observation = self.observations[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_observation = self.next_observations[idx]
        terminal = self.terminals[idx]

        # Debug prints
        # print(f"Reward at index {idx}: {reward}")

        return (
            torch.tensor(observation, dtype=torch.float32),
            torch.tensor(
                action,
                dtype=torch.long,
            ),  # actions are discrete in CartPole -> PyTorch requires the tensors to be torch.long
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_observation, dtype=torch.float32),
            bool(terminal),
        )
