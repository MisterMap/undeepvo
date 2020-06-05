import torch
import kornia


class InverseDepthSmoothnessLoss(torch.nn.Module):
    def __init__(self, lambda_depth=1.0):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.inverse_depth_smoothness_loss = kornia.losses.InverseDepthSmoothnessLoss()

    def forward(self, left_current_depth, left_current_image, right_current_depth, right_current_image):
        left_loss = self.inverse_depth_smoothness_loss(1.0 / left_current_depth, left_current_image)
        right_loss = self.inverse_depth_smoothness_loss(1.0 / right_current_depth, right_current_image)

        return self.lambda_depth * (left_loss + right_loss)
