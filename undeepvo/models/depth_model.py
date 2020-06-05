import torch
from torch import nn


class UnetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out_before_pooling = self.convs(x)
        out = self.maxpool(out_before_pooling)

        return out, out_before_pooling


class UnetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, x_bridge):
        x_up = self.upsample(x)
        x_concat = torch.cat([x_up, x_bridge], dim=1)
        out = self.convs(x_concat)

        return out


class LastUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, 1, kernel_size=1, padding=0),
            #nn.ReLU(),
        )

    def forward(self, x):
        out = self.convs(x)

        return out


class DepthNet(nn.Module):
    def __init__(self, n_base_channels=32, max_depth=100, min_depth=1, inverse_sigmoid=False):
        super().__init__()

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.inverse_sigmoid = inverse_sigmoid

        self.down_blocks = nn.ModuleList([
            UnetDownBlock(3, n_base_channels, kernel_size=7),  # 32 out
            UnetDownBlock(n_base_channels, n_base_channels * 2, kernel_size=5),  # 64
            UnetDownBlock(n_base_channels * 2, n_base_channels * 4),  # 128
            UnetDownBlock(n_base_channels * 4, n_base_channels * 8),  # 256
            UnetDownBlock(n_base_channels * 8, n_base_channels * 16),  # 512
            UnetDownBlock(n_base_channels * 16, n_base_channels * 16),  # 512
            UnetDownBlock(n_base_channels * 16, n_base_channels * 16)  # 512
        ])
        self.up_blocks = nn.ModuleList([
            UnetUpBlock(n_base_channels * 16, n_base_channels * 16),  # 512 out
            UnetUpBlock(n_base_channels * 16, n_base_channels * 8),  # 256
            UnetUpBlock(n_base_channels * 8, n_base_channels * 4),  # 128
            UnetUpBlock(n_base_channels * 4, n_base_channels * 2),  # 64
            UnetUpBlock(n_base_channels * 2, n_base_channels),  # 32
            UnetUpBlock(n_base_channels, n_base_channels // 2),  # 32
        ])

        self.last_up = LastUpBlock(n_base_channels // 2, 32)
#        self._last_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):

        out = x
        outputs_before_pooling = []
        before_pooling = None
        for (i, block) in enumerate(self.down_blocks):
            (out, before_pooling) = block(out)
            outputs_before_pooling.append(before_pooling)
        out = before_pooling

        for (i, block) in enumerate(self.up_blocks):
            out = block(out, outputs_before_pooling[-i - 2])

        out = self.last_up(out)
        #out = self._last_conv(out)

        if not self.inverse_sigmoid:
            out = self.min_depth + torch.sigmoid(out) * (self.max_depth - self.min_depth)
        else:
            b = 1 / self.min_depth
            a = 1 / self.max_depth - 1 / self.min_depth
            out = 1 / (a * torch.sigmoid(out) + b)

        return out
