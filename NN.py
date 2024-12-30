import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import h5py
from datetime import datetime

labels_to_index = {}

class HandLandmarksDataset(Dataset):
    def __init__(self, hdf5_file, split='train'):
        global labels_to_index
        self.samples = []
        self.labels = []

        # Open the HDF5 file
        with h5py.File(hdf5_file, 'r') as f:
            self.samples = f[f'{split}_samples'][:]
            self.labels = f[f'{split}_labels'][:]

        # Load labels_to_index mapping if it exists
        if os.path.exists("labels_to_index.json"):
            with open("labels_to_index.json", "r") as f:
                labels_to_index = json.load(f)
        else:
            # Create a mapping from labels to indices
            unique_labels = sorted(set(self.labels))
            labels_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            with open("labels_to_index.json", "w") as f:
                json.dump(labels_to_index, f, indent=4)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        landmarks = torch.tensor(self.samples[idx], dtype=torch.float32)
        label = self.labels[idx].decode('utf-8')  # Decode bytes to string if necessary
        label_index = labels_to_index[label]
        return landmarks, label_index

# Define the Neural Network
class HandGestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HandGestureNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Load datasets
train_dataset = HandLandmarksDataset('dataset.h5', split='train')
test_dataset = HandLandmarksDataset('dataset.h5', split='test')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create the model
input_size = 21 * 3  # 21 landmarks, each with x, y, z
num_classes = len(labels_to_index)
model = HandGestureNet(input_size, num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

train_accuracies = []

start_time = datetime.now()

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    train_accuracies.append(accuracy)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}%")
    scheduler.step(running_loss)

# Print final training accuracy
print(f"Final Training Accuracy: {accuracy:.4f}%")

end_time = datetime.now()

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.4f}%")

# Save the model if it performs better than the previous best
log_file = "accuracy_log.json"
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        log_data = json.load(f)
        if accuracy > log_data[-1]["max_train_accuracy"]:
            log_data[-1]["max_train_accuracy"] = accuracy
        if test_accuracy > log_data[-1]["max_test_accuracy"]:
            log_data[-1]["max_test_accuracy"] = test_accuracy
else:
    log_data = []
    log_data.append({
        "max_train_accuracy": accuracy,
        "max_test_accuracy": test_accuracy,
    })

max_train_accuracy = log_data[-1]["max_train_accuracy"]
max_test_accuracy = log_data[-1]["max_test_accuracy"]

save_model = False
if accuracy >= max_train_accuracy or test_accuracy >= max_test_accuracy:
    save_model = True
    # Save the model in ONNX format
    dummy_input = torch.randn(1, 63)  
    torch.onnx.export(model, dummy_input, "model.onnx")
    print("Model saved to model.onnx")

# Log the results
log_entry = {
    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
    "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
    "train_accuracy": accuracy,
    "test_accuracy": test_accuracy,
    "saved_model": save_model,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "other_techniques": 'ReduceLROnPlateau',
    "nb_epochs": epochs,
    "batch_size": 32,
    "nb_hidden_layers": 2,
    "hidden_layer_sizes": [128, 64],
}

log_data.append(log_entry)

with open(log_file, "w") as f:
    json.dump(log_data, f, indent=4)

print(f"Training log saved to {log_file}")

# Plot the training accuracies
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True)
plt.show()