import h5py
import numpy as np


class OfflineDataset:
    """Offline Dataset for loading pre-collected transitions."""

    def __init__(self, file_path):
        """Initialize the dataset by loading the HDF5 file."""
        self.file = h5py.File(file_path, "r")
        print("Shape of observations before squeezing:", np.array(self.file["observations"]).shape)
        # if statement for different dataset shapes
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

    def get_batch(self, batch_size):
        """Get a random batch of data."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.terminals[indices],
        )

    def get_batches(self, batch_size):
        """Generator that yields batches of experiences sequentially."""
        for i in range(0, self.size, batch_size):
            yield (
                self.observations[i : i + batch_size],
                self.actions[i : i + batch_size],
                self.rewards[i : i + batch_size],
                self.next_observations[i : i + batch_size],
                self.terminals[i : i + batch_size],
            )
