from torch import nn

from .depth_model import DepthNet
from .depth_net_res_net import DepthNetResNet
from .pose_model import PoseNet, PoseNetResNet
from .utils import init_weights


class UnDeepVO(nn.Module):
    def __init__(self, max_depth=100, min_depth=1, resnet=False, inverse_sigmoid=False):
        super(UnDeepVO, self).__init__()

        if not resnet:
            self.depth_net = DepthNet(max_depth=max_depth, min_depth=min_depth, inverse_sigmoid=inverse_sigmoid)
            self.pose_net = PoseNet()
            self.apply(init_weights)
        else:
            self.pose_net = PoseNetResNet()
            self.depth_net = DepthNetResNet(max_depth=max_depth, min_depth=min_depth, inverse_sigmoid=inverse_sigmoid)

    def depth(self, x):
        out = self.depth_net(x)
        return out

    def pose(self, x, reference_frame):
        (out_rotation, out_translation) = self.pose_net(x, reference_frame)
        return out_rotation, out_translation

    def forward(self, x, reference_frame):
        return self.depth(x), self.pose(x, reference_frame)
