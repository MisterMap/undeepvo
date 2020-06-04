import torch
from torch import nn
from torchvision import models

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

class UnetUpBlockResNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
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
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.convs(x)

        return out


class DepthNet(nn.Module):
    def __init__(self, n_base_channels=32, max_depth=None, min_depth=None, last_sigmoid=True):
        super().__init__()

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.last_sigmoid = last_sigmoid
        
        self.down_blocks = nn.ModuleList([
            UnetDownBlock(3, n_base_channels, kernel_size=7), # 32 out
            UnetDownBlock(n_base_channels, n_base_channels * 2, kernel_size=5), # 64
            UnetDownBlock(n_base_channels * 2, n_base_channels * 4), # 128
            UnetDownBlock(n_base_channels * 4, n_base_channels * 8), # 256
            UnetDownBlock(n_base_channels * 8, n_base_channels * 16),  # 512
            UnetDownBlock(n_base_channels * 16, n_base_channels * 16),  # 512
             UnetDownBlock(n_base_channels * 16, n_base_channels * 16)  # 512
        ])
        self.up_blocks = nn.ModuleList([
            UnetUpBlock(n_base_channels * 16, n_base_channels * 16), # 512 out
            UnetUpBlock(n_base_channels * 16, n_base_channels * 8), # 256 
            UnetUpBlock(n_base_channels * 8, n_base_channels * 4), # 128
            UnetUpBlock(n_base_channels * 4, n_base_channels * 2), # 64 
            UnetUpBlock(n_base_channels * 2, n_base_channels), # 32 
            UnetUpBlock(n_base_channels, n_base_channels // 2), # 32
        ])

        self.last_up = LastUpBlock(n_base_channels // 2, 32)
        self._last_conv = nn.Conv2d(32, 1, 1)
              
    def forward(self, x):

        out = x
        outputs_before_pooling = []
        for (i, block) in enumerate(self.down_blocks):
            (out, before_pooling) = block(out)
            outputs_before_pooling.append(before_pooling)
        out = before_pooling

        for (i, block) in enumerate(self.up_blocks):
            out = block(out, outputs_before_pooling[-i - 2])

        out = self.last_up(out)
        out = self._last_conv(out)
        if (self.max_depth != None) and (self.min_depth != None):
            if self.last_sigmoid:
                out = self.min_depth + torch.sigmoid(out) * (self.max_depth - self.min_depth)
            else:
                out = self.min_depth + (out + 1) * (self.max_depth - self.min_depth) / 2.
                out = torch.clamp(out, self.min_depth, self.max_depth)
        else:
            out = 1 / ((-9.99 * torch.sigmoid(out)) + 10) # from monodepth2

        return out
    
    
class DepthNetResNet(nn.Module):
    def __init__(self, n_base_channels=32, max_depth=None, min_depth=None, pretrained=True):
        super().__init__()

        self.max_depth = max_depth
        self.min_depth = min_depth

        self.skip_zero = nn.Conv2d(3, n_base_channels * 2, kernel_size=1, padding=0)
        self.resnet_part = list(models.resnet18(pretrained=pretrained).children())

        self.first_level = nn.Sequential(*self.resnet_part[:3])
        self.first_skip = nn.Sequential(
            nn.Conv2d(in_channels=n_base_channels * 2, out_channels=n_base_channels * 2, kernel_size=1, padding=0), # 64 -- 64
            nn.ReLU(inplace=True),
            )
        
        self.second_level = nn.Sequential(*self.resnet_part[3:5])
        self.second_skip = nn.Sequential(
            nn.Conv2d(in_channels=n_base_channels * 2, out_channels=n_base_channels * 2, kernel_size=1, padding=0),  # 64 -- 64
            nn.ReLU(inplace=True),
            )
        
        self.down_blocks = nn.ModuleList([
            UnetDownBlockResNet(n_base_channels * 4, n_base_channels * 4, self.resnet_part, 5), # 64 -- 128
            UnetDownBlockResNet(n_base_channels * 8, n_base_channels * 8, self.resnet_part, 6), # 128 -- 256
            UnetDownBlockResNet(n_base_channels * 16, n_base_channels * 16, self.resnet_part, 7), # 256 -- 512
        ])

        self.up_blocks = nn.ModuleList([
            UnetUpBlockResNet(n_base_channels * 8 + n_base_channels * 16, n_base_channels * 16), # 256 + 512 -- 512 
            UnetUpBlockResNet(n_base_channels * 4 + n_base_channels * 16, n_base_channels * 8), # 128 + 512 --- 256 
            UnetUpBlockResNet(n_base_channels * 2 + n_base_channels * 8, n_base_channels * 8), # 64 + 256 --- 256 
            UnetUpBlockResNet(n_base_channels * 2 + n_base_channels * 8, n_base_channels * 4), # 64 + 256 --- 128 
            UnetUpBlockResNet(n_base_channels * 2 + n_base_channels * 4, 32) # 64 + 128 --- 1
        ])

        # possible option
        # self.upsample = nn.ConvTranspose2d(n_base_channels * 4, n_base_channels * 4, 3, stride=2, padding=1) # nn.Upsample(scale_factor=2)
        # self.last_up = LastUpBlock(n_base_channels * 4, 1)
              
    def forward(self, x):
        
        outputs_before_pooling = []

        skip_0 = self.skip_zero(x)
        outputs_before_pooling.append(skip_0)

        x = self.first_level(x)
        skip_1 = self.first_skip(x)
        outputs_before_pooling.append(skip_1) # 64

        x = self.second_level(x)
        skip_2 = self.second_skip(x)
        outputs_before_pooling.append(skip_2) # 64
        out = x
        
        for (i, block) in enumerate(self.down_blocks):
            (out, before_pooling) = block(out)
            outputs_before_pooling.append(before_pooling)

        out = before_pooling

        for (i, block) in enumerate(self.up_blocks):
            out = block(out, outputs_before_pooling[-i - 2])

        if (self.max_depth != None) and (self.min_depth != None):
            out = self.min_depth + torch.sigmoid(out) * (self.max_depth - self.min_depth) 
        else:
            out = 1 / ((-9.99 * torch.sigmoid(out)) + 10) # from monodepth2

        return out
