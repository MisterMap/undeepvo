from torch import nn

from .depth_model import DepthNet, DepthNetResNet
from .pose_model import PoseNet, PoseNetResNet
from .utils import init_weights


class UnDeepVO(nn.Module):
    def __init__(self, max_depth=None, min_depth=None, resnet=False):
        super(UnDeepVO, self).__init__()
        
        if resnet == False:
            self.depth_net = DepthNet(max_depth=max_depth, min_depth=min_depth)
            self.pose_net = PoseNet()
            self.apply(init_weights)
        else:
            self.pose_net = PoseNetResNet()
            self.depth_net = DepthNetResNet(max_depth=max_depth, min_depth=min_depth)
        
    def depth(self, x):
        out = self.depth_net(x)
        return out

    def pose(self, x):
        (out_rotation, out_translation) = self.pose_net(x)
        return out_rotation, out_translation

    def forward(self, x):
        return self.depth(x), self.pose(x)
