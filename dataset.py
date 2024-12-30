import os
import json
import torch
import h5py
import numpy as np

def labels_to_index(labels):
    unique_labels = sorted(set(labels))
    labels_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    with open("labels_to_index.json", "w") as f:
        json.dump(labels_to_index, f, indent=4)

def process_directory(directory):
    samples = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Parse JSON and add samples
            with open(file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    landmarks = torch.tensor(item["landmarks"], dtype=torch.float32).view(-1)
                    samples.append(landmarks.numpy())
                    label = item["label"]
                    labels.append(label)
    return samples, labels

def create_hdf5_dataset(train_dir, test_dir, output_file, labels_to_index=False):
    train_samples, train_labels = process_directory(train_dir)
    test_samples, test_labels = process_directory(test_dir)

    if labels_to_index:
        labels_to_index(train_labels)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('train_samples', data=train_samples)
        f.create_dataset('train_labels', data=np.string_(train_labels))
        f.create_dataset('test_samples', data=test_samples)
        f.create_dataset('test_labels', data=np.string_(test_labels))

# Define directories
train_dir = "processed_coordinates/train"
test_dir = "processed_coordinates/test"
output_file = "dataset.h5"

# Create the HDF5 dataset
create_hdf5_dataset(train_dir, test_dir, output_file)
print(f"Dataset HDF5 created at {output_file}")