import h5py
import numpy as np


class OfflineDataset:
    """Offline Dataset for loading pre-collected transitions."""

    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")

    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self.file["observations"]), size=batch_size)
        return (
            self.file["observations"][indices],
            self.file["actions"][indices],
            self.file["rewards"][indices],
            self.file["next_observations"][indices],
            self.file["terminals"][indices],
        )
