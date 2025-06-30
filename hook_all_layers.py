import torch
import torch.nn as nn

def register_all_hooks(model):
    layer_io = {}  # 存储每层的输入输出
    handles = []   # 存储 hook 句柄，便于之后 remove

    def hook_fn(module, input, output):
        layer_io[module] = {
            'input_shape': input[0].shape if isinstance(input, tuple) else None,
            'output_shape': output.shape if isinstance(output, torch.Tensor) else None
        }

    # 给所有子模块注册 hook（排除 model 自身）
    for name, layer in model.named_modules():
        if layer != model:
            handle = layer.register_forward_hook(hook_fn)
            handles.append(handle)

    return layer_io, handles

'''
# 定义你的模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 创建模型
model = MyModel()

# 注册 hook，获取每层输入输出
layer_io, handles = register_all_hooks(model)

# 执行一次前向传播
x = torch.randn(1, 3, 32, 32)
out = model(x)

# 打印所有层的输入输出 shape
for layer, io in layer_io.items():
    print(f"\nLayer: {layer.__class__.__name__}")
    print(f"Input shape: {io['input_shape']}")
    print(f"Output shape: {io['output_shape']}")

# 清理 hook，避免影响后续运行
for h in handles:
    h.remove()
'''