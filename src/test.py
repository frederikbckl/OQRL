import h5py

file_path = "offline_cartpole_test_v5.hdf5"

with h5py.File(file_path, "r") as f:
    print(f.keys())  # Prints all top-level keys in the HDF5 file
