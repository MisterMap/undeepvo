import torch
import kornia


class DisparityConsistencyLoss(torch.nn.Module):
    def __init__(self, Bf, left_camera_matrix, right_camera_matrix, transform_from_left_to_right,
                 lambda_disparity=0.85):
        super().__init__()
        self.Bf = Bf

        self.left_camera_matrix = left_camera_matrix
        self.right_camera_matrix = right_camera_matrix
        self.transform_from_left_to_right = transform_from_left_to_right

        self.lambda_disparity = lambda_disparity

        self.l1_loss = torch.nn.L1Loss()

    def get_disparities(self, left_depth_map, right_depth_map):
        left_disparity = self.Bf / left_depth_map
        right_disparity = self.Bf / right_depth_map
        return left_disparity, right_disparity

    def generate_disparity_maps(self, left_disparity, right_disparity, left_current_depth, right_current_depth):
        generated_right_disparity = kornia.warp_frame_depth(image_src=left_disparity,
                                                            depth_dst=right_current_depth,
                                                            src_trans_dst=self.transform_from_left_to_right,
                                                            camera_matrix=self.left_camera_matrix)

        generated_left_disparity = kornia.warp_frame_depth(image_src=right_disparity,
                                                           depth_dst=left_current_depth,
                                                           src_trans_dst=torch.inverse(
                                                               self.transform_from_left_to_right),
                                                           camera_matrix=self.right_camera_matrix)

        return generated_left_disparity, generated_right_disparity

    def forward(self, left_current_depth, right_current_depth):
        left_disparity, right_disparity = self.get_disparities(left_current_depth, right_current_depth)

        generated_left_disparity, generated_right_disparity = self.generate_disparity_maps(left_disparity,
                                                                                           right_disparity,
                                                                                           left_current_depth,
                                                                                           right_current_depth)

        return self.lambda_disparity * self.l1_loss(left_disparity,
                                                    generated_left_disparity) + self.lambda_disparity * self.l1_loss(
            right_disparity, generated_right_disparity)
