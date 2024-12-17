import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Hyperparameters
batch_size = 64
knn_k = 10
hidden_dim = 64
epochs = 5
learning_rate = 0.001

# Dataset (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# KNN Model
class KNN:
    def __init__(self, k):
        self.k = k
        self.train_data = None
        self.train_labels = None

    def fit(self, data, labels):
        self.train_data = data
        self.train_labels = labels

    def predict(self, x):
        # Compute L2 distances
        distances = torch.cdist(x, self.train_data)
        # Get the indices of the k the smallest distances
        knn_indices = distances.topk(self.k, largest=False).indices
        # Gather the labels of these neighbors
        knn_labels = self.train_labels[knn_indices]
        # Return the most common label
        return knn_labels.mode(dim=1).values

# MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.fc2(x)
        return x

# Train MLP
def train_mlp(model, loader, criterion, optimizer):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate Models
def evaluate_mlp(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def evaluate_knn(knn, loader):
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.view(images.size(0), -1), labels
        predictions = knn.predict(images)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
    return correct / total

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train and Evaluate KNN
train_features = train_dataset.data.float().view(-1, 28*28) / 255.0
train_labels = train_dataset.targets
test_features = test_dataset.data.float().view(-1, 28*28) / 255.0
test_labels = test_dataset.targets

knn = KNN(knn_k)
knn.fit(train_features, train_labels)

knn_accuracy = evaluate_knn(knn, DataLoader(test_dataset, batch_size=256))
print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")

# Train and Evaluate MLP
mlp = MLP(input_dim=28*28, hidden_dim=hidden_dim, output_dim=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train_mlp(mlp, train_loader, criterion, optimizer)
    print(f"Epoch {epoch+1}/{epochs} complete.")

mlp_accuracy = evaluate_mlp(mlp, test_loader)
print(f"MLP Accuracy: {mlp_accuracy * 100:.2f}%")
