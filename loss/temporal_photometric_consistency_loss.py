import torch
import kornia

class TemporalPhotometricConsistencyLoss(torch.nn.Module):
    def __init__(self, camera_matrix, transformation_matrix, lambda_s, window_size = 11, reduction: str = "none", max_val: float = 1.0):
        super().__init__()
        self.camera_matrix = camera_matrix
        self.inverse_camera_matrix = torch.inverse(self.camera_matrix)
        self.transformation_matrix = transformation_matrix
        self.inverse_transformation_matrix = torch.inverse(self.transformation_matrix)

        self.lambda_s = lambda_s
        self.SSIM_loss = kornia.SSIM(window_size = window_size, reduction = reduction, max_val = max_val)
        self.l1_loss = torch.nn.L1Loss()

    def generate_next_image(self, image_previous, depth_previous):
        generated_next_image = self.camera_matrix @ self.transformation_matrix @ depth_previous @ self.inverse_camera_matrix @ image_previous
        return generated_next_image

    def generate_previous_image(self, image_next, depth_next):
        generated_previous_image = self.camera_matrix @ self.inverse_transformation_matrix @ depth_next @ self.inverse_camera_matrix @ image_next
        return generated_previous_image

    def forward(self, current_image, next_image, current_depth, next_depth):
        generated_image_next = self.generate_next_image(current_image, current_depth)
        generated_image_previos = self.generate_previous_image(next_image, next_depth)

        loss_current = self.lambda_s * -self.SSIM_loss(generated_image_previos, current_image) + (
                    1 - self.lambda_s) * self.l1_loss(generated_image_previos, current_image)

        loss_next = self.lambda_s * -self.SSIM_loss(generated_image_next, next_image) + (1 - self.lambda_s) * self.l1_loss(
            generated_image_next, next_image)

        return loss_current + loss_next
