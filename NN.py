import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

labels_to_index = {}

class HandLandmarksDataset(Dataset):
    def __init__(self, path, is_train=True):
        global labels_to_index
        self.samples = []

        if is_train:
            # Read all JSON files in the directory recursively
            for root, _, files in os.walk(path):
                for file in files:
                    label = os.path.splitext(file)[0]  # Use the filename as the label
                    file_path = os.path.join(root, file)

                    # Parse JSON and add samples
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        for item in data:
                            landmarks = torch.tensor(item["landmarks"], dtype=torch.float32).view(-1)
                            self.samples.append((landmarks, label))
            # Create a mapping from labels to indices
            labels = sorted(set(label for _, label in self.samples))
            labels_to_index = {label: idx for idx, label in enumerate(labels)}
        else:
            # Load test data from a single JSON file
            with open(path, 'r') as f:
                data = json.load(f)
                for item in data:
                    landmarks = torch.tensor(item["landmarks"], dtype=torch.float32).view(-1)
                    label = item["label"]  # Use the "label" field
                    self.samples.append((landmarks, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        global labels_to_index
        landmarks, label = self.samples[idx]
        label_idx = labels_to_index[label]
        return landmarks, label_idx

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

# Dataset paths
data_directory = "processed_coordinates"
train_directory = os.path.join(data_directory, "train")
test_file = os.path.join(data_directory, "test", "test.json")

# Load datasets
train_dataset = HandLandmarksDataset(train_directory, is_train=True)
test_dataset = HandLandmarksDataset(test_file, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create the model
input_size = 21 * 3  # 21 landmarks, each with x, y, z
num_classes = len(labels_to_index)
model = HandGestureNet(input_size, num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_accuracies = []
test_accuracies = []

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Training loop
epochs = 50
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
test_accuracies.append(test_accuracy)
print(f"Test Accuracy: {test_accuracy:.4f}%")

# Plot the training accuracies
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True)
plt.show()