import torch


class PoseLoss(torch.nn.Module):
    def __init__(self, lambda_position, lambda_angle):
        super().__init__()
        self.lambda_position = lambda_position
        self.lambda_angle = lambda_angle

    # TODO: add matrix calcualtion for cameras
    def forward(self, left_position, right_position,
                left_angle, right_angle):
        l1_loss = torch.nn.L1Loss()
        return self.lambda_position * l1_loss(left_position, right_position) + self.lambda_angle * l1_loss(left_angle,
                                                                                                           right_angle)