import kornia
import torch


class TemporalPhotometricConsistencyLoss(torch.nn.Module):
    def __init__(self, left_camera_matrix, right_camera_matrix,
                 lambda_s=0.85, window_size=11, reduction: str = "mean",
                 max_val: float = 1.0):
        super().__init__()
        self.left_camera_matrix = left_camera_matrix
        self.right_camera_matrix = right_camera_matrix
        self.inverse_left_camera_matrix = torch.inverse(self.left_camera_matrix)
        self.inverse_right_camera_matrix = torch.inverse(self.right_camera_matrix)

        self.lambda_s = lambda_s
        self.ssim_loss = kornia.losses.SSIM(window_size=window_size, reduction=reduction, max_val=max_val)
        self.l1_loss = torch.nn.L1Loss()

    def generate_transformation(self, current_position, next_position, current_angle, next_angle):
        current_rot_matrix = kornia.geometry.angle_axis_to_rotation_matrix(
            current_angle)  # current transformtaiton matrix

        current_position = current_position.unsqueeze(dim=2)

        current_transform_matrix = torch.cat((current_rot_matrix, current_position), dim=2)
        tmp = torch.tensor([[0, 0, 0, 1]] * current_position.shape[0], dtype=torch.float)
        tmp = tmp.unsqueeze(dim=1)
        current_transform_matrix = torch.cat((current_transform_matrix, tmp), dim=1)

        next_rot_matrix = kornia.geometry.angle_axis_to_rotation_matrix(next_angle)  # next transformation matrix

        next_position = next_position.unsqueeze(dim=2)
        next_transform_matrix = torch.cat((next_rot_matrix, next_position), dim=2)
        tmp = torch.tensor([[0, 0, 0, 1]] * next_position.shape[0], dtype=torch.float)
        tmp = tmp.unsqueeze(dim=1)
        next_transform_matrix = torch.cat((next_transform_matrix, tmp), dim=1)

        return kornia.geometry.linalg.relative_transformation(current_transform_matrix, next_transform_matrix)

    def generate_next_image(self, image_previous, depth_next, transformation, camera_matrix):
        generated_next_image = kornia.warp_frame_depth(image_previous,
                                                       depth_next,
                                                       transformation,
                                                       camera_matrix)
        return generated_next_image

    def generate_previous_image(self, image_next, depth_previous, inverse_transformation, camera_matrix):
        generated_previous_image = kornia.warp_frame_depth(image_next,
                                                           depth_previous,
                                                           inverse_transformation,
                                                           camera_matrix)
        return generated_previous_image

    def forward(self, left_current_image, left_next_image, left_current_depth, left_next_depth,
                right_current_image, right_next_image, right_current_depth, right_next_depth,
                left_current_position, right_current_position, left_current_angle, right_current_angle,
                left_next_position, right_next_position, left_next_angle, right_next_angle):
        self.left_transformation_from_previous_to_next = self.generate_transformation(left_current_position,
                                                                                      left_next_position,
                                                                                      left_current_angle,
                                                                                      left_next_angle)
        self.inverse_left_transformation_from_previous_to_next = torch.inverse(
            self.left_transformation_from_previous_to_next)

        self.right_transformation_from_previous_to_next = self.generate_transformation(right_current_position,
                                                                                       right_next_position,
                                                                                       right_current_angle,
                                                                                       right_next_angle)
        self.inverse_right_transformation_from_previous_to_next = torch.inverse(
            self.right_transformation_from_previous_to_next)

        left_generated_next_image = self.generate_next_image(left_current_image, left_current_depth,
                                                             self.left_transformation_from_previous_to_next,
                                                             self.left_camera_matrix)

        left_generated_current_image = self.generate_previous_image(left_next_image, left_next_depth,
                                                                    self.inverse_left_transformation_from_previous_to_next,
                                                                    self.left_camera_matrix)

        left_loss_current = self.lambda_s * self.ssim_loss(left_generated_current_image, left_current_image) + \
                            (1 - self.lambda_s) * self.l1_loss(left_generated_current_image, left_current_image)

        left_loss_next = self.lambda_s * self.ssim_loss(left_generated_next_image, left_next_image) + \
                         (1 - self.lambda_s) * self.l1_loss(left_generated_next_image, left_next_image)

        right_generated_next_image = self.generate_next_image(right_current_image, right_current_depth,
                                                              self.right_transformation_from_previous_to_next,
                                                              self.right_camera_matrix)

        right_generated_current_image = self.generate_previous_image(right_next_image, right_next_depth,
                                                                     self.inverse_right_transformation_from_previous_to_next,
                                                                     self.right_camera_matrix)

        right_loss_current = self.lambda_s * self.ssim_loss(right_generated_current_image, right_current_image) + \
                             (1 - self.lambda_s) * self.l1_loss(right_generated_current_image, right_current_image)

        right_loss_next = self.lambda_s * self.ssim_loss(right_generated_next_image, right_next_image) + \
                          (1 - self.lambda_s) * self.l1_loss(right_generated_next_image, right_next_image)

        return left_loss_current + left_loss_next + right_loss_current + right_loss_next
