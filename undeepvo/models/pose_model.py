import torch
from torch import nn
from torchvision import models
        
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
        self.rot3 = nn.Linear(512, 4)

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

class PoseNetResNet(nn.Module):
    def __init__(self, n_base_channels=16, pretrained=True):
        super(PoseNetResNet, self).__init__()

        self.resnet_part = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()

        self.rot1 = nn.Linear(512 * 6 * 6, 512)
        self.rot2 = nn.Linear(512, 512)
        self.rot3 = nn.Linear(512, 4)

        self.transl1 = nn.Linear(512 * 6 * 6, 512)
        self.transl2 = nn.Linear(512, 512)
        self.transl3 = nn.Linear(512, 3)

    def forward(self, x):

        x = self.resnet_part(x)
        x = self.avgpool(x)
        out = self.flatten(x)

        out_rot = self.rot3(torch.relu(self.rot2(torch.relu(self.rot1(out)))))
        out_transl = self.transl3(torch.relu(self.transl2(torch.relu(self.transl1(out)))))

        return (out_rot, out_transl)

