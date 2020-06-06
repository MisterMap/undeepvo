import torch.nn as nn

from undeepvo.data.cameras_calibration import CamerasCalibration
from undeepvo.utils import ResultDataPoint
from .losses import SpatialLosses, TemporalImageLosses
from .pose_metric import PoseMetric


class UnsupervisedCriterion(nn.Module):
    def __init__(self, cameras_calibration: CamerasCalibration, lambda_position, lambda_angle, lambda_s,
                 lambda_disparity=0.01, lambda_registration=0.01, lambda_smoothness=1.0):
        super(UnsupervisedCriterion, self).__init__()
        self.spatial_losses = SpatialLosses(cameras_calibration.camera_baseline,
                                            cameras_calibration.focal_length,
                                            cameras_calibration.left_camera_matrix,
                                            cameras_calibration.right_camera_matrix,
                                            cameras_calibration.transform_from_left_to_right,
                                            lambda_position, lambda_angle, lambda_s, lambda_disparity,
                                            lambda_smoothness)

        self.temporal_losses = TemporalImageLosses(cameras_calibration.left_camera_matrix,
                                                   cameras_calibration.right_camera_matrix,
                                                   lambda_s, lambda_registration)
        self.pose_metric = PoseMetric()

    def forward(self, left_current_output: ResultDataPoint, right_current_output: ResultDataPoint,
                left_next_output: ResultDataPoint, right_next_output: ResultDataPoint):
        current_spatial_loss, current_photometric_loss, current_disparity_loss, current_inverse_depth_smoothness_loss, current_pose_loss = \
            self.spatial_losses(left_current_output.input_image, right_current_output.input_image,
                                left_current_output.depth, right_current_output.depth,
                                left_current_output.translation, right_current_output.translation,
                                left_current_output.rotation, right_current_output.rotation)
        next_spatial_loss, next_photometric_loss, next_disparity_loss, next_inverse_depth_smoothness_loss, next_pose_loss = \
            self.spatial_losses(left_next_output.input_image, right_next_output.input_image,
                                left_next_output.depth, right_next_output.depth,
                                left_next_output.translation, right_next_output.translation,
                                left_next_output.rotation, right_next_output.rotation)
        temporal_loss, registration_loss = self.temporal_losses(left_current_output.input_image,
                                                                left_next_output.input_image,
                                                                left_current_output.depth,
                                                                left_next_output.depth,
                                                                right_current_output.input_image,
                                                                right_next_output.input_image,
                                                                right_current_output.depth,
                                                                right_next_output.depth,
                                                                left_current_output.translation,
                                                                right_current_output.translation,
                                                                left_current_output.rotation,
                                                                right_current_output.rotation,
                                                                left_next_output.translation,
                                                                right_next_output.translation,
                                                                left_next_output.rotation,
                                                                right_next_output.rotation)
        return ((current_spatial_loss + next_spatial_loss) / 2 + temporal_loss + registration_loss,
                (current_photometric_loss + next_photometric_loss) / 2,
                (current_disparity_loss + next_disparity_loss) / 2,
                (current_inverse_depth_smoothness_loss + next_inverse_depth_smoothness_loss) / 2,
                (current_pose_loss + next_pose_loss) / 2,
                temporal_loss,
                registration_loss)

    def calculate_relative_pose_error(self, left_current_output: ResultDataPoint, right_current_output: ResultDataPoint,
                                      left_next_output: ResultDataPoint, right_next_output: ResultDataPoint,
                                      delta_position, delta_angle,
                                      inverse_delta_position, inverse_delta_angle):
        left_current_loss = self.pose_metric.calculate_relative_pose_error(left_current_output.translation,
                                                                           left_current_output.rotation,
                                                                           delta_position, delta_angle)
        right_current_loss = self.pose_metric.calculate_relative_pose_error(right_current_output.translation,
                                                                            right_current_output.rotation,
                                                                            delta_position, delta_angle)
        left_next_loss = self.pose_metric.calculate_relative_pose_error(left_next_output.translation,
                                                                        left_next_output.rotation,
                                                                        inverse_delta_position, inverse_delta_angle)
        right_next_loss = self.pose_metric.calculate_relative_pose_error(right_next_output.translation,
                                                                         right_next_output.rotation,
                                                                         inverse_delta_position, inverse_delta_angle)
        return (left_current_loss + right_current_loss + left_next_loss + right_next_loss) / 4