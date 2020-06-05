import kornia
import torch


class SpatialPhotometricConsistencyLoss(torch.nn.Module):
    def __init__(self, lambda_s, left_camera_matrix, right_camera_matrix, transform_from_left_to_right,
                 window_size=11, reduction: str = "none", max_val: float = 1.0):
        super().__init__()
        self.lambda_s = lambda_s
        self.window_size = window_size
        self.reduction = reduction
        self.max_val = max_val

        self.left_camera_matrix = left_camera_matrix
        self.right_camera_matrix = right_camera_matrix
        self.transform_from_left_to_right = transform_from_left_to_right

        self.l1_loss = torch.nn.L1Loss()
        self.SSIM_loss = kornia.losses.SSIM(window_size=self.window_size, reduction=self.reduction,
                                            max_val=self.max_val)

    def forward(self, left_current_img, right_current_img, left_current_depth, right_current_depth):
        generated_right_img = kornia.warp_frame_depth(image_src=left_current_img,
                                                      depth_dst=right_current_depth,
                                                      src_trans_dst=torch.inverse(self.transform_from_left_to_right),
                                                      camera_matrix=self.left_camera_matrix)

        generated_left_img = kornia.warp_frame_depth(image_src=right_current_img,
                                                     depth_dst=left_current_depth,
                                                     src_trans_dst=self.transform_from_left_to_right,
                                                     camera_matrix=self.right_camera_matrix)

        left_img_loss = self.lambda_s * self.SSIM_loss(generated_left_img, left_current_img) + \
                        (1 - self.lambda_s) * self.l1_loss(generated_left_img, left_current_img)
        right_img_loss = self.lambda_s * self.SSIM_loss(generated_right_img, right_current_img) + \
                         (1 - self.lambda_s) * self.l1_loss(generated_right_img, right_current_img)

        return (left_img_loss + right_img_loss) / 2
