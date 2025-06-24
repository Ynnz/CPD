import torch
import numpy as np

# 固定输入
inp = torch.tensor([[[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]]])  # shape: [1, 1, 3, 3]

# 固定卷积核（1 kernel, 1 in_channel, 2x2）
weight = torch.tensor([[[[1.0, 0.0],
                         [0.0, -1.0]]]])  # shape: [1, 1, 2, 2]

bias = torch.tensor([0.0])

# 定义卷积层（手动赋值）
conv = torch.nn.Conv2d(1, 1, kernel_size=2, bias=True)
with torch.no_grad():
    conv.weight.copy_(weight)
    conv.bias.copy_(bias)

out = conv(inp)
print("PyTorch output:\n", out.squeeze().numpy())

# 保存为 .npz 供 C 端读取
np.savez("test_fixed_conv.npz",
         input=inp.squeeze().numpy(),
         weight=weight.squeeze().numpy(),
         bias=bias.numpy(),
         ref=out.squeeze().numpy())
