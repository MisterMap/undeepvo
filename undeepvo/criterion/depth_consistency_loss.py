import torch
import kornia


class DepthConsistencyLoss(torch.nn.Module):
    def __init__(self, left_camera_matrix, right_camera_matrix, transform_from_left_to_right,
                 lambda_depth=0.85):
        super().__init__()

        #self.left_camera_matrix = left_camera_matrix
        #self.right_camera_matrix = right_camera_matrix
        #self.transform_from_left_to_right = transform_from_left_to_right

        self.lambda_depth = lambda_depth

    def generate_depth_maps(self, left_current_depth, right_current_depth):
        generated_right_depth = kornia.warp_frame_depth(image_src=left_current_depth,
                                                        depth_dst=right_current_depth,
                                                        src_trans_dst=self.transform_from_left_to_right,
                                                        camera_matrix=self.left_camera_matrix)

        generated_left_depth = kornia.warp_frame_depth(image_src=right_current_depth,
                                                       depth_dst=left_current_depth,
                                                       src_trans_dst=torch.inverse(
                                                           self.transform_from_left_to_right),
                                                       camera_matrix=self.right_camera_matrix)

        return generated_left_depth, generated_right_depth

    def forward(self, left_current_depth, left_current_image, right_current_depth, right_current_image):
        left_loss = kornia.inverse_depth_smoothness_loss(1 / left_current_depth, left_current_image)
        right_loss = kornia.inverse_depth_smoothness_loss(1 / right_current_depth, right_current_image)

        return self.lambda_depth * (left_loss + right_loss) / 2.0
