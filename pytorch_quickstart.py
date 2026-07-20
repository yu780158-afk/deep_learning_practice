import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#下载训练集与测试集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print(f"训练集大小：{len(training_data)}")
print(f"测试集大小：{len(test_data)}")
print(f"单张图片形状：{training_data[0][0].shape}")

#创建 DataLoader
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

print(f"训练集分批数：{len(train_dataloader)}")
print(f"测试集分批数：{len(test_dataloader)}")
print(f"每批形状：{next(iter(train_dataloader))[0].shape}")

# 定义神经网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()                    # 把 28×28 拉平成 784 个数字
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),                     # 第1层：784 → 512
            nn.ReLU(),                                  # 激活函数
            nn.Linear(512, 512),                       # 第2层：512 → 512
            nn.ReLU(),
            nn.Linear(512, 10),                        # 输出层：512 → 10（10个类别）
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 创建模型实例，放到 GPU 上
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)
print(f"模型在: {device}")
print(model)

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()           # 分类任务的损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # 随机梯度下降

# 训练函数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 前向传播：让模型猜
        pred = model(X)
        # 计算猜得有多离谱
        loss = loss_fn(pred, y)

        # 反向传播：调整参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每 100 批打印一次
        if batch % 100 == 0:
            loss_val = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

# 测试函数
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"测试准确率: {(100*correct):>0.1f}%, 平均loss: {test_loss:>8f}")

# 训练 5 轮
epochs = 5
for t in range(epochs):
    print(f"\n-------- 第 {t+1} 轮 --------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("\n训练完成！✅")
