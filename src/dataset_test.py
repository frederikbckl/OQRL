# import h5py
from dataset import OfflineDataset

# with h5py.File("offline_cartpole_v2.hdf5", "r") as f:
#     for key in f.keys():
#         print(f"Dataset: {key}, Type: {type(f[key])}, Shape: {f[key].shape}")

dataset = OfflineDataset("offline_cartpole_v2.hdf5")

print("Observations shape:", dataset.observations.shape)
print("Actions shape:", dataset.actions.shape)
print("Rewards shape:", dataset.rewards.shape)
print("Next Observations shape:", dataset.next_observations.shape)
print("Terminals shape:", dataset.terminals.shape)
