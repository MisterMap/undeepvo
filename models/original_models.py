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
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
    def forward(self, x):

        out = self.convs(x)
        
        return out

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
class Unet(nn.Module):
    def __init__(self, n_base_channels=32):
        super().__init__()
        
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

        self.last_up = LastUpBlock(n_base_channels // 2, 1)
              
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

        return out
        
class PoseNet(nn.Module):
    def __init__(self, n_base_channels=16):
        super(PoseNet, self).__init__()

        self.vgg_part = nn.ModuleList([
            VggBlock(in_channels=3, out_channels=n_base_channels, kernel_size=7, padding=3, maxpool=False), # 16
            VggBlock(in_channels=n_base_channels, out_channels=n_base_channels, kernel_size=7, padding=3, maxpool=True), # 16
            VggBlock(in_channels=n_base_channels, out_channels=n_base_channels*2, kernel_size=5, padding=2, maxpool=False), # 32
            VggBlock(in_channels=n_base_channels*2, out_channels=n_base_channels*2, kernel_size=5, padding=2, maxpool=True), # 32
            VggBlock(in_channels=n_base_channels*2, out_channels=n_base_channels*4, kernel_size=3, padding=1, maxpool=False), # 64
            VggBlock(in_channels=n_base_channels*4, out_channels=n_base_channels*4, kernel_size=3, padding=1, maxpool=True), # 64
            VggBlock(in_channels=n_base_channels*4, out_channels=n_base_channels*8, kernel_size=3, padding=1, maxpool=False), # 128
            VggBlock(in_channels=n_base_channels*8, out_channels=n_base_channels*8, kernel_size=3, padding=1, maxpool=True), # 128
            VggBlock(in_channels=n_base_channels*8, out_channels=n_base_channels*16, kernel_size=3, padding=1, maxpool=False), # 256
            VggBlock(in_channels=n_base_channels*16, out_channels=n_base_channels*16, kernel_size=3, padding=1, maxpool=True), # 256
            VggBlock(in_channels=n_base_channels*16, out_channels=n_base_channels*16, kernel_size=3, padding=1, maxpool=False), # 256
            VggBlock(in_channels=n_base_channels*16, out_channels=n_base_channels*16, kernel_size=3, padding=1, maxpool=True), # 256
            VggBlock(in_channels=n_base_channels*16, out_channels=n_base_channels*32, kernel_size=3, padding=1, maxpool=False), # 512
            VggBlock(in_channels=n_base_channels*32, out_channels=n_base_channels*32, kernel_size=3, padding=1, maxpool=True), # 512
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()

        self.rot1 = nn.Linear(n_base_channels*32*7*7, 512)
        self.rot2 = nn.Linear(512, 512)
        self.rot3 = nn.Linear(512, 3)

        self.transl1 = nn.Linear(n_base_channels*32*7*7, 512)
        self.transl2 = nn.Linear(512, 512)
        self.transl3 = nn.Linear(512, 3)

    def forward(self, x):

        for block in self.vgg_part:
            x = block(x)

        x = self.avgpool(x)
        out = self.flatten(x)

        out_rot = self.rot3(self.rot2(self.rot1(out)))
        out_transl = self.transl3(self.transl2(self.transl1(out)))

        return (out_rot, out_transl)

class VggBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, maxpool):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        if maxpool == True:
            self.conv.add_module("max_pool", nn.MaxPool2d(kernel_size=2, stride=2))
        
    def forward(self, x):

        out = self.conv(x)
        
        return out
