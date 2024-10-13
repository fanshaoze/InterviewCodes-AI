import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define a simple dataset
torch.manual_seed(42)
n_samples = 1000
n_features = 20
X = torch.randn(n_samples, n_features)
y = torch.randn(n_samples, 1)

# Define three models: No normalization, Layer Normalization, and Batch Normalization
class NoNormalizationModel(nn.Module):
    def __init__(self, n_features):
        super(NoNormalizationModel, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc11 = nn.Linear(128, 128)
        self.fc12 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc11(x))
        x = torch.relu(self.fc12(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LayerNormModel(nn.Module):
    def __init__(self, n_features):
        super(LayerNormModel, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc11 = nn.Linear(128, 128)
        self.ln11 = nn.LayerNorm(128)
        self.fc12 = nn.Linear(128, 128)
        self.ln12 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln11(self.fc11(x)))
        x = torch.relu(self.ln12(self.fc12(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x

class BatchNormModel(nn.Module):
    def __init__(self, n_features):
        super(BatchNormModel, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc11 = nn.Linear(128, 128)
        self.bn11 = nn.BatchNorm1d(128)
        self.fc12 = nn.Linear(128, 128)
        self.bn12 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn11(self.fc11(x)))
        x = torch.relu(self.bn12(self.fc12(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Initialize models
models = {
    "No Normalization": NoNormalizationModel(n_features),
    "Layer Normalization": LayerNormModel(n_features),
    "Batch Normalization": BatchNormModel(n_features)
}

# Training setup
criterion = nn.MSELoss()
n_epochs = 150
learning_rate = 0.001

# Store loss for each model
loss_history = {key: [] for key in models.keys()}

# Training loop for each model
for model_name, model in models.items():
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        loss_history[model_name].append(loss.item())
        if epoch % 10 == 0:
            print(f"{model_name} - Epoch {epoch}, Loss: {loss.item()}")

# Plot the loss curves
plt.figure(figsize=(10, 6))
for model_name, losses in loss_history.items():
    plt.plot(losses, label=model_name)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Norm Comparison: No vs. Layer vs. Batch, 4 layers')
plt.legend()
plt.grid()
# plt.show()
plt.savefig('loss_comparison_4_layers.png')
