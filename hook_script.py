import torch
import torch.nn as nn

def analyze_model(model, input_tensor):
    # Store all information here
    layer_info = []

    # For capturing input/output during forward pass
    def hook_fn(module, input, output):
        layer_info.append({
            'name': name_lookup.get(module),
            'type': type(module).__name__,
            'input_shape': input[0].shape if isinstance(input, tuple) else None,
            'output_shape': output.shape if isinstance(output, torch.Tensor) else None,
            'has_weight': hasattr(module, 'weight') and module.weight is not None,
            'weight_shape': tuple(module.weight.shape) if hasattr(module, 'weight') and module.weight is not None else None,
            'has_bias': hasattr(module, 'bias') and module.bias is not None,
            'bias_shape': tuple(module.bias.shape) if hasattr(module, 'bias') and module.bias is not None else None
        })

    # Build name lookup: module object -> name
    name_lookup = {m: n for n, m in model.named_modules()}

    # Register hooks on leaf layers only
    handles = []
    for module in model.modules():
        if len(list(module.children())) == 0:  # leaf only
            h = module.register_forward_hook(hook_fn)
            handles.append(h)

    # Forward pass
    _ = model(input_tensor)

    # Remove hooks
    for h in handles:
        h.remove()

    return layer_info

'''
Example usage:
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleNet()
dummy_input = torch.randn(1, 3, 32, 32)

info = analyze_model(model, dummy_input)

# Print the results
for layer in info:
    print(layer)

    
Optional: Export to CSV or JSON
import pandas as pd
df = pd.DataFrame(info)
df.to_csv("layer_summary.csv", index=False)
'''