import torch
import torch.nn as nn

# 1. 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 2. 创建模型对象
model = MyModel()

# 3. 定义 hook 函数
def hook_fn(module, input, output):
    print(f"层: {module.__class__.__name__}")
    print(f"输入: {input[0].shape}")
    print(f"输出: {output.shape}")

# 4. 注册 hook（这行你问的代码就在这里）
handle = model.conv1.register_forward_hook(hook_fn)

# 5. 执行前向传播（这时 hook_fn 会被自动调用）
x = torch.randn(1, 3, 32, 32)
out = model(x)

# 6. 可选：移除 hook（防止干扰后续操作）
handle.remove()
