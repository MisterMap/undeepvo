import torch
from torch import nn
from torchvision import models


class PoseNet(nn.Module):
    def __init__(self, n_base_channels=16):
        super(PoseNet, self).__init__()

        self.vgg_part = nn.ModuleList([
            VggBlock(in_channels=6, out_channels=n_base_channels, kernel_size=7, padding=3, maxpool=False),  # 16
            VggBlock(in_channels=n_base_channels, out_channels=n_base_channels, kernel_size=7, padding=3, maxpool=True),
            # 16
            VggBlock(in_channels=n_base_channels, out_channels=n_base_channels * 2, kernel_size=5, padding=2,
                     maxpool=False),  # 32
            VggBlock(in_channels=n_base_channels * 2, out_channels=n_base_channels * 2, kernel_size=5, padding=2,
                     maxpool=True),  # 32
            VggBlock(in_channels=n_base_channels * 2, out_channels=n_base_channels * 4, kernel_size=3, padding=1,
                     maxpool=False),  # 64
            VggBlock(in_channels=n_base_channels * 4, out_channels=n_base_channels * 4, kernel_size=3, padding=1,
                     maxpool=True),  # 64
            VggBlock(in_channels=n_base_channels * 4, out_channels=n_base_channels * 8, kernel_size=3, padding=1,
                     maxpool=False),  # 128
            VggBlock(in_channels=n_base_channels * 8, out_channels=n_base_channels * 8, kernel_size=3, padding=1,
                     maxpool=True),  # 128
            VggBlock(in_channels=n_base_channels * 8, out_channels=n_base_channels * 16, kernel_size=3, padding=1,
                     maxpool=False),  # 256
            VggBlock(in_channels=n_base_channels * 16, out_channels=n_base_channels * 16, kernel_size=3, padding=1,
                     maxpool=True),  # 256
            VggBlock(in_channels=n_base_channels * 16, out_channels=n_base_channels * 16, kernel_size=3, padding=1,
                     maxpool=False),  # 256
            VggBlock(in_channels=n_base_channels * 16, out_channels=n_base_channels * 16, kernel_size=3, padding=1,
                     maxpool=True),  # 256
            VggBlock(in_channels=n_base_channels * 16, out_channels=n_base_channels * 32, kernel_size=3, padding=1,
                     maxpool=False),  # 512
            VggBlock(in_channels=n_base_channels * 32, out_channels=n_base_channels * 32, kernel_size=3, padding=1,
                     maxpool=True),  # 512
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()

        self.rot1 = nn.Linear(n_base_channels * 32 * 7 * 7, 512)
        self.rot2 = nn.Linear(512, 512)
        self.rot3 = nn.Linear(512, 3)

        self.transl1 = nn.Linear(n_base_channels * 32 * 7 * 7, 512)
        self.transl2 = nn.Linear(512, 512)
        self.transl3 = nn.Linear(512, 3)

    def forward(self, target_frame, reference_frame):
        x = torch.cat([target_frame, reference_frame], dim=1)
        for block in self.vgg_part:
            x = block(x)

        x = self.avgpool(x)
        out = self.flatten(x)

        out_rot = 0.01 * self.rot3(torch.relu(self.rot2(torch.relu(self.rot1(out)))))
        out_transl = 0.01 * self.transl3(torch.relu(self.transl2(torch.relu(self.transl1(out)))))

        return out_rot, out_transl


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
    def __init__(self, n_base_channels=16, pretrained=True, input_images=2):
        super(PoseNetResNet, self).__init__()
        self._first_layer = nn.Conv2d(3 * input_images, 64, kernel_size=(7, 7), stride=(2, 2),
                                      padding=(3, 3), bias=False)
        resnet = models.resnet18(pretrained=pretrained)
        self.resnet_part = nn.Sequential(*list(resnet.children())[1:-2])
        if pretrained:
            loaded_weights = resnet.state_dict()["conv1.weight"]
            loaded_weights = torch.cat([loaded_weights] * input_images, 1) / input_images
            self._first_layer.load_state_dict({"weight": loaded_weights})

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()

        self.rot1 = nn.Linear(512 * 6 * 6, 512)
        self.rot2 = nn.Linear(512, 512)
        self.rot3 = nn.Linear(512, 3)

        self.transl1 = nn.Linear(512 * 6 * 6, 512)
        self.transl2 = nn.Linear(512, 512)
        self.transl3 = nn.Linear(512, 3)

    def forward(self, target_frame, reference_frame):
        x = torch.cat([target_frame, reference_frame], dim=1)
        x = self._first_layer(x)
        x = self.resnet_part(x)
        x = self.avgpool(x)
        out = self.flatten(x)

        out_rot = 0.01 * self.rot3(torch.relu(self.rot2(torch.relu(self.rot1(out)))))
        out_transl = 0.01 * self.transl3(torch.relu(self.transl2(torch.relu(self.transl1(out)))))

        return (out_rot, out_transl)
