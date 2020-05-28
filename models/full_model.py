from .depth_model import DepthNet
from .pose_model import PoseNet
from .utils import init_weights
import torch
from torch import nn

class UnDeepVO(nn.Module):

    def __init__(self):
        super(UnDeepVO, self).__init__()

        self.depth_net = DepthNet()
        self.pose_net = PoseNet()
        self.apply(init_weights)

    def depth(self, x):

        out = self.depth_net(x)

        return out

    def pose(self, x):

        (out_rot, out_transl) = self.pose_net(x)

        return (out_rot, out_transl)
