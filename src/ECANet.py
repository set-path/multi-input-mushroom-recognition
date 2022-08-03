import torch
import torch.nn as nn


class ECANet(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ECANet, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=(
            kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # inputs:(batch_size, channels, height, width)
        batch_size, channels, _, _ = inputs.size()
        # y:(batch_size, channels, 1, 1)
        y = self.avg_pool(inputs)
        # y:(batch_size, channels)
        y = self.conv(y.squeeze(-1).transpose(-2, -1))
        y = self.sigmoid(y).transpose(-2, -1).unsqueeze(-1)
        return inputs*y
