import torch


class CamerasCalibration(object):
    def __init__(self, camera_baseline, left_camera_matrix, right_camera_matrix, device="cuda:0"):
        self.camera_baseline = camera_baseline
        self.left_camera_matrix = torch.from_numpy(left_camera_matrix).to(device)[None].float()
        self.right_camera_matrix = torch.from_numpy(right_camera_matrix).to(device)[None].float()
        self.focal_length = self.left_camera_matrix[0, 0, 0]
        self.transform_from_left_to_right = torch.tensor(((1, 0, 0, 0),
                                                          (0, 1, 0, 0),
                                                          (0, 0, 1, 0),
                                                          (0, 0, 0, 1)))[None].to(device).float()
        self.transform_from_left_to_right[0, 0, 3] = -self.camera_baseline
