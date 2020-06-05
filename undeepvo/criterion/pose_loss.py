import kornia
import torch

from undeepvo.utils.math import generate_transformation, generate_pose


class PoseLoss(torch.nn.Module):
    def __init__(self, lambda_position, lambda_angle, right_from_left_transformation):
        super().__init__()
        self._lambda_position = lambda_position
        self._lambda_angle = lambda_angle
        self._right_from_left_transformation = right_from_left_transformation
        self._l1_loss = torch.nn.L1Loss()

    def forward(self, left_position, right_position,
                left_angle, right_angle):
        right_transformation = generate_transformation(right_position, right_angle)
        right_transformed_position, right_transformed_angle = generate_pose(
            kornia.compose_transformations(right_transformation, self._right_from_left_transformation)
        )
        translation_loss = self._lambda_position * self._l1_loss(left_position, right_transformed_position)
        rotation_loss = self._lambda_angle * self._l1_loss(left_angle, right_transformed_angle)
        return translation_loss + rotation_loss
