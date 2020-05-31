import torch
import kornia

from .spatial_photometric_consistency_loss import SpatialPhotometricConsistencyLoss
from .disparity_consistency_loss import DisparityConsistencyLoss
from .pose_loss import PoseLoss

from .temporal_photometric_consistency_loss import TemporalPhotometricConsistencyLoss

class Criterion(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

class SpatialLosses(torch.nn.Module):
    def __init__(self, camera_baseline, focal_length, left_camera_matrix, right_camera_matrix, transfrom_from_left_to_right, lambda_position, lambda_angle,
                 lambda_s, window_size = 11, reduction: str = "none", max_val: float = 1.0):
        super().__init__()
        self.baseline = camera_baseline
        self.focal_length = focal_length
        self.Bf = self.baseline * self.focal_length

        self.left_camera_matrix = left_camera_matrix
        self.right_camera_matrix = right_camera_matrix
        self.transfrom_from_left_to_right = transfrom_from_left_to_right

        self.lambda_position = lambda_position
        self.lambda_angle = lambda_angle

        self.lambda_s = lambda_s
        self.window_size = window_size
        self.reduction = reduction
        self.max_val = max_val

        self.photometric_consistency_loss = SpatialPhotometricConsistencyLoss(self.lambda_s, self.left_camera_matrix, self.right_camera_matrix, self.transfrom_from_left_to_right,
                 window_size = self.window_size, reduction = self.reduction, max_val = self.max_val)
        self.disparity_consistency_loss = DisparityConsistencyLoss(self.Bf, self.left_camera_matrix, self.right_camera_matrix, self.transfrom_from_left_to_right)
        self.pose_loss = PoseLoss(self.lambda_position, self.lambda_angle)

    def forward(self, left_current_image, right_current_image,
                left_current_depth, right_current_depth,
                lambda_position, lambda_angle, left_position, right_position, left_angle, right_angle,
                ):
        photometric_consistency_loss = self.photometric_consistency_loss(left_current_image, right_current_image, left_current_depth, right_current_depth,
                                          self.transfrom_from_left_to_right, self.left_camera_matrix, self.right_camera_matrix)

        disparity_consistency_loss = self.disparity_consistency_loss(left_current_depth, right_current_depth)
        pose_loss = self.pose_loss(self.lambda_position, self.lambda_angle, left_position, right_position, left_angle, right_angle)
        return photometric_consistency_loss + disparity_consistency_loss + pose_loss


class TemporalImageLosses(torch.nn.Module):
    def __init__(self,camera_matrix, transformation_matrix):
        super().__init__()
        self.camera_matrix = camera_matrix
        self.transformation_matrix = transformation_matrix

        #self.geometric_registration_loss = ThreeDGeometricRegistrationLoss(self.transformation_matrix)
        self.photometric_consistency_loss = TemporalPhotometricConsistencyLoss(self.camera_matrix, self.transformation_matrix)

    def forward(self, image_previous, depth_previous, image_next, depth_next, point_cloud_previous, point_cloud_next, regulazation):
        self.geometric_registration_loss(point_cloud_previous, point_cloud_next)
        self.photometric_consistency_loss(image_previous, image_next, depth_previous, depth_next, regulazation)