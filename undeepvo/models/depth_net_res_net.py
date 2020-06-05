import torch
from torch import nn
from torchvision import models


class UnetUpBlockResNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, padding_mode='reflect'),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, padding_mode='reflect'),
            nn.ELU(),
        )
        
    def forward(self, x, x_bridge):
        x_up = self.upsample(x)
        x_concat = torch.cat([x_up, x_bridge], dim=1)
        out = self.convs(x_concat)

        return out
    
class LastUpBlockResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU()
        )
        
    def forward(self, x):

        out = self.convs(x)
        
        return out


class UnetDownBlockResNet(nn.Module):
    def __init__(self, in_channels, out_channel, resnet_modules, resnet_level):
        super().__init__()

        self.convs = nn.Sequential(*resnet_modules[resnet_level])
        self.skip_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.convs(x)
        out_skip = self.skip_layer(out)

        return out, out_skip


class DepthNetResNet(nn.Module):
    def __init__(self, n_base_channels=32, max_depth=100., min_depth=1., pretrained=True, inverse_sigmoid=False):
        super().__init__()

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.inverse_sigmoid = inverse_sigmoid

        self.skip_zero = nn.Conv2d(3, n_base_channels * 2, kernel_size=3, padding=1)
        self.skip_zero1 = nn.Conv2d(n_base_channels * 2, n_base_channels * 2, kernel_size=3, padding=1)
        
        self.resnet_part = list(models.resnet18(pretrained=pretrained).children())

        self.first_level = nn.Sequential(*self.resnet_part[:3])
        self.first_skip = nn.Sequential(
            nn.Conv2d(in_channels=n_base_channels * 2, out_channels=n_base_channels * 2, kernel_size=1, padding=0),
            # 64 -- 64
            nn.ReLU(inplace=True),
        )

        self.second_level = nn.Sequential(*self.resnet_part[3:5])
        self.second_skip = nn.Sequential(
            nn.Conv2d(in_channels=n_base_channels * 2, out_channels=n_base_channels * 2, kernel_size=1, padding=0),
            # 64 -- 64
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.ModuleList([
            UnetDownBlockResNet(n_base_channels * 4, n_base_channels * 4, self.resnet_part, 5),  # 64 -- 128
            UnetDownBlockResNet(n_base_channels * 8, n_base_channels * 8, self.resnet_part, 6),  # 128 -- 256
            UnetDownBlockResNet(n_base_channels * 16, n_base_channels * 16, self.resnet_part, 7),  # 256 -- 512
        ])

        self.up_blocks = nn.ModuleList([
            UnetUpBlockResNet(n_base_channels * 8 + n_base_channels * 16, n_base_channels * 16),  # 256 + 512 -- 512
            UnetUpBlockResNet(n_base_channels * 4 + n_base_channels * 16, n_base_channels * 8),  # 128 + 512 --- 256
            UnetUpBlockResNet(n_base_channels * 2 + n_base_channels * 8, n_base_channels * 8),  # 64 + 256 --- 256
            UnetUpBlockResNet(n_base_channels * 2 + n_base_channels * 8, n_base_channels * 4),  # 64 + 256 --- 128
            UnetUpBlockResNet(n_base_channels * 2 + n_base_channels * 4, 64)  # 64 + 128 --- 1
        ])
        
        self.last_up = LastUpBlockResNet(n_base_channels * 2, 32)
        
        self._last_conv = nn.Conv2d(32, 1, 1)
        # possible option
        # self.upsample = nn.ConvTranspose2d(n_base_channels * 4, n_base_channels * 4, 3, stride=2, padding=1)
        # nn.Upsample(scale_factor=2)
        # self.last_up = LastUpBlock(n_base_channels * 4, 1)

    def forward(self, x):

        outputs_before_pooling = []

        skip_0 = torch.relu(self.skip_zero(x))
        skip_00 = torch.relu(self.skip_zero1(skip_0))
        outputs_before_pooling.append(skip_00)

        x = self.first_level(x)
        skip_1 = self.first_skip(x)
        outputs_before_pooling.append(skip_1)  # 64

        x = self.second_level(x)
        skip_2 = self.second_skip(x)
        outputs_before_pooling.append(skip_2)  # 64
        out = x
        before_pooling = None
        for (i, block) in enumerate(self.down_blocks):
            (out, before_pooling) = block(out)
            outputs_before_pooling.append(before_pooling)

        out = before_pooling

        for (i, block) in enumerate(self.up_blocks):
            out = block(out, outputs_before_pooling[-i - 2])
            
        out = self.last_up(out)
        out = self._last_conv(out)
        
        if not self.inverse_sigmoid:
            out = self.min_depth + torch.sigmoid(out) * (self.max_depth - self.min_depth)
        else:
            b = 1 / self.min_depth
            a = 1 / self.max_depth - 1 / self.min_depth
            out = 1 / (a * torch.sigmoid(out) + b)

        return out
