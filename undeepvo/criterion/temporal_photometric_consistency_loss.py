import kornia
import torch

from undeepvo.utils.math import generate_transformation


class TemporalPhotometricConsistencyLoss(torch.nn.Module):
    def __init__(self, camera_matrix, right_camera_matrix,
                 lambda_s=0.85, window_size=11, reduction: str = "mean",
                 max_val: float = 1.0):
        super().__init__()
        self.camera_matrix = camera_matrix
        self.lambda_s = lambda_s
        self.ssim_loss = kornia.losses.SSIM(window_size=window_size, reduction=reduction, max_val=max_val)
        self.l1_loss = torch.nn.L1Loss()

    def calculate_loss(self, image1, image2):
        loss = self.lambda_s * self.ssim_loss(image1, image2) + (1 - self.lambda_s) * self.l1_loss(image1, image2)
        return loss

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

    def forward(self, current_image, next_image, current_depth, next_depth,
                current_position, current_angle, next_position, next_angle):
        transformation_from_next_to_current = generate_transformation(current_position, current_angle)
        transformation_from_current_to_next = generate_transformation(next_position, next_angle)

        generated_next_image = self.generate_next_image(current_image, next_depth,
                                                        transformation_from_next_to_current)

        generated_current_image = self.generate_current_image(next_image, current_depth,
                                                              transformation_from_current_to_next)

        next_loss = self.calculate_loss(generated_next_image, next_image)
        current_loss = self.calculate_loss(generated_current_image, current_image)
        return (next_loss + current_loss) / 2
