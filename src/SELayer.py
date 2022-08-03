import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction_coefficient=16):
        super(SELayer, self).__init__()
        assert in_channels % reduction_coefficient == 0, 'error'
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction_coefficient),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_coefficient, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        out = self.avg_pool(x).view(batch_size, channels)
        out = self.fc1(out).view(batch_size, channels, 1, 1)
        return x*out
