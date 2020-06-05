import torch

from .disparity_consistency_loss import DisparityConsistencyLoss
from .inverse_depth_smoothness_loss import InverseDepthSmoothnessLoss
from .pose_loss import PoseLoss
from .registration_loss import GeometricRegistrationLoss
from .spatial_photometric_consistency_loss import SpatialPhotometricConsistencyLoss
from .temporal_photometric_consistency_loss import TemporalPhotometricConsistencyLoss


class SpatialLosses(torch.nn.Module):
    def __init__(self, camera_baseline, focal_length, left_camera_matrix, right_camera_matrix,
                 transform_from_left_to_right, lambda_position, lambda_angle,
                 lambda_s=0.85, lambda_disparity=0.85, lambda_smoothness=0.85, window_size=11, reduction: str = "mean",
                 max_val: float = 1.0):
        super().__init__()
        self.baseline = camera_baseline
        self.focal_length = focal_length
        self.Bf = self.baseline * self.focal_length

        self.left_camera_matrix = left_camera_matrix
        self.right_camera_matrix = right_camera_matrix
        self.transform_from_left_to_right = transform_from_left_to_right

        self.lambda_position = lambda_position
        self.lambda_angle = lambda_angle

        self.lambda_s = lambda_s
        self.window_size = window_size
        self.reduction = reduction
        self.max_val = max_val

        self.photometric_consistency_loss = SpatialPhotometricConsistencyLoss(self.lambda_s, self.left_camera_matrix,
                                                                              self.right_camera_matrix,
                                                                              self.transform_from_left_to_right,
                                                                              window_size=self.window_size,
                                                                              reduction=self.reduction,
                                                                              max_val=self.max_val)
        self.disparity_consistency_loss = DisparityConsistencyLoss(self.Bf, self.left_camera_matrix,
                                                                   self.right_camera_matrix,
                                                                   self.transform_from_left_to_right, lambda_disparity)
        self.inverse_depth_smoothness_loss = InverseDepthSmoothnessLoss(lambda_smoothness)

        self.pose_loss = PoseLoss(self.lambda_position, self.lambda_angle,
                                  self.transform_from_left_to_right)

    def forward(self, left_current_image, right_current_image,
                left_current_depth, right_current_depth,
                left_position, right_position, left_angle, right_angle):
        photometric_consistency_loss = self.photometric_consistency_loss(left_current_image, right_current_image,
                                                                         left_current_depth, right_current_depth)
        disparity_consistency_loss = self.disparity_consistency_loss(left_current_depth, right_current_depth)
        inverse_depth_smoothness_loss = self.inverse_depth_smoothness_loss(left_current_depth, left_current_image,
                                                                           right_current_depth, right_current_image)
        pose_loss = self.pose_loss(left_position, right_position, left_angle, right_angle)
        return (photometric_consistency_loss + disparity_consistency_loss + inverse_depth_smoothness_loss + pose_loss,
                photometric_consistency_loss,
                disparity_consistency_loss, inverse_depth_smoothness_loss, pose_loss)


class TemporalImageLosses(torch.nn.Module):
    def __init__(self, left_camera_matrix, right_camera_matrix,
                 lambda_s=0.85, registration_lambda=0.1):
        super().__init__()
        self.left_camera_matrix = left_camera_matrix
        self.right_camera_matrix = right_camera_matrix

        self.lambda_s = lambda_s

        self.left_temporal_photometric_loss = TemporalPhotometricConsistencyLoss(self.left_camera_matrix, self.lambda_s)
        self.right_temporal_photometric_loss = TemporalPhotometricConsistencyLoss(self.right_camera_matrix,
                                                                                  self.lambda_s)
        self.registration_loss = GeometricRegistrationLoss(registration_lambda, self.left_camera_matrix)

    def forward(self, left_current_image, left_next_image, left_current_depth, left_next_depth,
                right_current_image, right_next_image, right_current_depth, right_next_depth,
                left_current_position, right_current_position, left_current_angle, right_current_angle,
                left_next_position, right_next_position, left_next_angle, right_next_angle):
        left_temporal_photometric_loss = self.left_temporal_photometric_loss(left_current_image,
                                                                             left_next_image,
                                                                             left_current_depth,
                                                                             left_next_depth,
                                                                             left_current_position,
                                                                             left_current_angle,
                                                                             left_next_position,
                                                                             left_next_angle)
        right_temporal_photometric_loss = self.right_temporal_photometric_loss(right_current_image,
                                                                               right_next_image,
                                                                               right_current_depth,
                                                                               right_next_depth,
                                                                               right_current_position,
                                                                               right_current_angle,
                                                                               right_next_position,
                                                                               right_next_angle)
        left_registration_loss = self.registration_loss(left_current_depth,
                                                        left_next_depth,
                                                        left_current_position,
                                                        left_next_position,
                                                        left_current_angle,
                                                        left_next_angle)
        right_registration_loss = self.registration_loss(right_current_depth,
                                                         right_next_depth,
                                                         right_current_position,
                                                         right_next_position,
                                                         right_current_angle,
                                                         right_next_angle)
        return ((left_temporal_photometric_loss + right_temporal_photometric_loss) / 2,
                (left_registration_loss + right_registration_loss) / 2)
