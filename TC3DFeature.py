import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), stride=1, use_se=True):
        super().__init__()
        self.use_se = use_se
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.residual = nn.Identity()

        if use_se:
            self.se = SpatialTemporalSE(out_channels)
        else:
            self.se = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + res
        x = F.relu(x)
        x = self.se(x)
        return x


class SpatialTemporalSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, C, D, H, W)
        y = x.mean(dim=[2,3,4])  
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(x.size(0), x.size(1), 1, 1, 1)


class TCConv3DFeatureExtractor(nn.Module):
    def __init__(self, in_channels=4, base_channels=16, num_blocks=3, use_se=True):
        super().__init__()
        blocks = []
        channels = in_channels
        for i in range(num_blocks):
            # stride=2
            stride = 1 if i == 0 else 2
            blocks.append(Conv3DResidualBlock(channels, base_channels*(2**i), stride=stride, use_se=use_se))
            channels = base_channels*(2**i)
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self, x):
        # x: (B, 4, 9, 201, 201)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # (B, C)
        return x
