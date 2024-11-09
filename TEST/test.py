import torch
import torch.nn as nn

# 定义简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 创建模型实例
model = SimpleNN()

# 将模型移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 创建输入张量并移动到GPU
input_data = torch.randn(10).to(device)

# 前向传播
output = model(input_data)
print(output)
