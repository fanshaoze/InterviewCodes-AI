import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 创建一个简单的MLP模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# 初始化方法
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

# 训练模型并记录损失
def train_model(init_fn=None):
    model = SimpleMLP()
    if init_fn:
        model.apply(init_fn)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 用随机数据进行训练
    inputs = torch.randn(64, 10)
    targets = torch.randn(64, 1)

    losses = []
    for epoch in range(2000):
        print(epoch)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# 训练不同初始化方法的模型
loss_default = train_model()  # 默认初始化
loss_xavier = train_model(xavier_init)  # Xavier初始化
loss_kaiming = train_model(kaiming_init)  # Kaiming初始化

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(loss_default, label='Default Init', color='blue', lw=2.5)
plt.plot(loss_xavier, label='Xavier Init', color='green', lw=2.5)
plt.plot(loss_kaiming, label='Kaiming Init', color='red', lw=2.5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Comparison of Initialization Methods')
plt.ylim(0, 2)
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('initialization_compare.png')
