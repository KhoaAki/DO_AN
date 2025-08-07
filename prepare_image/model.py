import torch
import torch.nn as nn
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.InstanceNorm2d(features, affine=True))  # thay thế BatchNorm
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn(x)
    
def load_model(checkpoint_path, channels=1, num_of_layers=25, device='cpu'):
    model = DnCNN(channels=channels, num_of_layers=num_of_layers).to(device)
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    # Gỡ bỏ prefix "module." nếu có
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")  # cần vì huấn luyện dùng nn.DataParallel
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model