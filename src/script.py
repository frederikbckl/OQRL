import h5py
import numpy as np

file_path = "offline_cartpole_test_v5.hdf5"

with h5py.File(file_path, "r") as f:
    rewards = np.array(f["rewards"])
    print("Rewards type:", type(rewards))
    print("Rewards shape:", rewards.shape)
    print("First few rewards:", rewards[:10])  # Print a few sample values
