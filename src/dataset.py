import h5py
import numpy as np


class OfflineDataset:
    """Offline Dataset for loading pre-collected transitions."""

    def __init__(self, file_path, rng=None):
        """Initialize the dataset by loading the HDF5 file."""
        try:
            self.file = h5py.File(file_path, "r")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
        self.rng = rng or np.random.default_rng()

        # print(
        #     "\nShape of observations before squeezing:",
        #     np.array(self.file["observations"]).shape,
        # )

        # if statement necessary for different dataset shapes
        if np.array(self.file["observations"]).shape == (100000, 4):
            self.observations = np.array(self.file["observations"])  # Shape (100000, 4)
            self.actions = np.array(self.file["actions"])  # Adjust if needed
            self.rewards = np.array(self.file["rewards"])  # Already 1D
            self.next_observations = np.array(self.file["next_observations"])  # Shape (100000, 4)
        else:
            # Convert to NumPy arrays and remove extra dimensions using np.squeeze
            self.observations = np.squeeze(
                np.array(self.file["observations"]),
                axis=1,
            )  # Shape (174700, 4)
            self.actions = np.squeeze(
                np.array(self.file["actions"]),
                axis=(1, 2),
            )  # Shape (174700,)
            self.rewards = np.array(self.file["rewards"])  # Already 1D, no need for squeeze
            self.next_observations = np.squeeze(
                np.array(self.file["next_observations"]),
                axis=1,
            )  # Shape (174700, 4)
        self.terminals = np.array(self.file["terminals"])  # Already 1D, no need for squeeze
        self.size = len(self.observations)

    def __iter__(self):
        """Make the dataset iterable."""
        for i in range(self.size):
            yield (
                self.observations[i],
                self.actions[i],
                self.rewards[i],
                self.next_observations[i],
                self.terminals[i],
            )

    def sample(self, size):
        """Sample a seeded subset of the dataset."""
        indices = self.rng.choice(self.size, size=size, replace=False)

        observations = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_observations = self.next_observations[indices]
        terminals = self.terminals[indices]

        return list(zip(observations, actions, rewards, next_observations, terminals))
