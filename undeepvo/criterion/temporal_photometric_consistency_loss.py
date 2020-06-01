import kornia
import torch


class TemporalPhotometricConsistencyLoss(torch.nn.Module):
    def __init__(self, camera_matrix, transformation_matrix, lambda_s=0.85, window_size=11, reduction: str = "none",
                 max_val: float = 1.0):
        super().__init__()
        self.camera_matrix = camera_matrix
        self.inverse_camera_matrix = torch.inverse(self.camera_matrix)
        self.transformation_matrix = transformation_matrix
        self.inverse_transformation_matrix = torch.inverse(self.transformation_matrix)

        self.lambda_s = lambda_s
        self.ssim_loss = kornia.losses.SSIM(window_size=window_size, reduction=reduction, max_val=max_val)
        self.l1_loss = torch.nn.L1Loss()

    def generate_next_image(self, image_previous, depth_previous):
        image_previous = image_previous[0]
        self.inverse_camera_matrix = self.inverse_camera_matrix[0]
        print("inverse_camera_matrix.shape", self.inverse_camera_matrix.shape)
        print("image_previous.shape = ", image_previous.shape)
        depth_previous = depth_previous
        print("self.inverse_camera_matrix @ image_previous", torch.matmul(self.inverse_camera_matrix, image_previous))
        generated_next_image = self.camera_matrix @ self.transformation_matrix @ \
                               depth_previous @ self.inverse_camera_matrix @ image_previous
        return generated_next_image

    def generate_previous_image(self, image_next, depth_next):
        image_next = image_next[0]
        depth_next = depth_next[0]
        generated_previous_image = self.camera_matrix @ self.inverse_transformation_matrix @ \
                                   depth_next @ self.inverse_camera_matrix @ image_next
        return generated_previous_image

    def forward(self, current_image, next_image, current_depth, next_depth):
        generated_image_next = self.generate_next_image(current_image, current_depth)
        print(f"generated_image_next = {generated_image_next}")
        generated_image_previous = self.generate_previous_image(next_image, next_depth)
        print(f"generated_image_previous = {generated_image_previous}")

        loss_current = self.lambda_s * self.ssim_loss(generated_image_previous, current_image) + \
                       (1 - self.lambda_s) * self.l1_loss(generated_image_previous, current_image)

        loss_next = self.lambda_s * self.ssim_loss(generated_image_next, next_image) + \
                    (1 - self.lambda_s) * self.l1_loss(generated_image_next, next_image)

        return loss_current + loss_next
