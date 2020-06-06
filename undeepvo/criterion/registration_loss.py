import torch
import torch.nn as nn

from undeepvo.utils.math import generate_relative_transformation
import kornia


class GeometricRegistrationLoss(torch.nn.Module):
    def __init__(self, registration_lambda, camera_matrix):
        super().__init__()
        self._loss = nn.L1Loss()
        self._registration_lambda = registration_lambda
        self.camera_matrix = camera_matrix

    def generate_next_image(self, current_image, next_depth, transformation_from_next_to_current):
        generated_next_image = kornia.warp_frame_depth(current_image,
                                                       next_depth,
                                                       transformation_from_next_to_current,
                                                       self.camera_matrix)
        return generated_next_image

    def generate_current_image(self, next_image, current_depth, transformation_from_current_to_next):
        generated_current_image = kornia.warp_frame_depth(next_image,
                                                          current_depth,
                                                          transformation_from_current_to_next,
                                                          self.camera_matrix)
        return generated_current_image

    def forward(self, current_depth, next_depth, current_position, next_position, current_angle, next_angle):
        transformation_from_current_to_next = generate_relative_transformation(current_position,
                                                                               current_angle,
                                                                               next_position,
                                                                               next_angle)
        transformation_from_next_to_current = generate_relative_transformation(next_position,
                                                                               next_angle,
                                                                               current_position,
                                                                               current_angle)
        generated_next_depth = self.generate_next_image(current_depth, next_depth,
                                                        transformation_from_next_to_current)

        generated_current_depth = self.generate_current_image(next_depth, current_depth,
                                                              transformation_from_current_to_next)

        loss_previous = self._loss(generated_current_depth, current_depth)
        loss_next = self._loss(generated_next_depth, next_depth)

        return (loss_previous + loss_next) / 2 * self._registration_lambda
