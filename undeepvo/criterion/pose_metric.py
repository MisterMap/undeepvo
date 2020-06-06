from undeepvo.utils.math import generate_relative_transformation
import torch


class PoseMetric(object):
    def calculate_relative_pose_error(self, predicted_delta_pose, predicted_delta_angle,
                                      truth_delta_pose, truth_delta_angle):
        error_transform = generate_relative_transformation(predicted_delta_pose, predicted_delta_angle,
                                                           truth_delta_pose, truth_delta_angle)
        return self.translation_error(error_transform)

    @staticmethod
    def translation_error(error_transform):
        translation = error_transform[:, :3, 3]
        return torch.sqrt(translation[:, 0] ** 2 + translation[:, 1] ** 2 + translation[:,  2] ** 2)