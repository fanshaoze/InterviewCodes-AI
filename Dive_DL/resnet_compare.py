import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定义一个没有残差模块的简单CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义一个包含残差模块的CNN
class ResNetCNN(nn.Module):
    def __init__(self):
        super(ResNetCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.res_block = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16)
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        residual = x
        x = self.res_block(x)
        x += residual  # 添加残差连接
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 定义训练函数
def train_model(model, optimizer, criterion, epochs=30):
    model.train()
    loss_list = []
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(trainloader)
        loss_list.append(average_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')
    return loss_list

# 实例化模型、损失函数和优化器
simple_cnn = SimpleCNN()
resnet_cnn = ResNetCNN()

criterion = nn.CrossEntropyLoss()
optimizer_simple = optim.Adam(simple_cnn.parameters(), lr=0.001)
optimizer_resnet = optim.Adam(resnet_cnn.parameters(), lr=0.001)

# 训练模型并记录损失
print("Training Simple CNN...")
loss_simple = train_model(simple_cnn, optimizer_simple, criterion)

print("\nTraining ResNet CNN...")
loss_resnet = train_model(resnet_cnn, optimizer_resnet, criterion)

# 绘制损失曲线

plt.figure(figsize=(10, 5))
plt.plot(loss_simple, label='Simple CNN', marker='o')
plt.plot(loss_resnet, label='ResNet CNN', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Comparison: Simple CNN vs. ResNet CNN')
plt.legend()
plt.grid()
# plt.show()
plt.savefig('resnet_compare.png')
plt.close()
