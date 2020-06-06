import os
import unittest

import numpy as np
import torch
import torchvision
import wget
from PIL import Image

from undeepvo.criterion import TemporalPhotometricConsistencyLoss
from undeepvo.data.cameras_calibration import CamerasCalibration
from test.utils import create_depth_map, read_calib
from undeepvo.utils.math import generate_relative_transformation
device = "cpu"


class TestTemporalLoss(unittest.TestCase):
    def setUp(self) -> None:
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        data_link = "http://vision.middlebury.edu/stereo/data/scenes2014/datasets/Adirondack-perfect"
        wget.download(f"{data_link}/im0.png", "tmp/im0.png")
        wget.download(f"{data_link}/im1.png", "tmp/im1.png")
        wget.download(f"{data_link}/calib.txt", "tmp/calib.txt")
        wget.download(f"{data_link}/disp0.pfm", "tmp/disp0.pfm")
        wget.download(f"{data_link}/disp1.pfm", "tmp/disp1.pfm")

        left_current_img = Image.open("tmp/im0.png")
        right_current_img = Image.open("tmp/im1.png")
        calib = read_calib("tmp/calib.txt")
        left_current_depth = create_depth_map("tmp/disp0.pfm", calib)
        right_current_depth = np.roll(create_depth_map("tmp/disp1.pfm", calib), -210, axis=1)
        transform = torchvision.transforms.ToTensor()
        self.left_current_img = transform(left_current_img)[None].to(device).float()
        self.right_current_img = transform(right_current_img)[None].to(device).float().roll(-210, dims=3)

        self.left_current_depth = transform(left_current_depth.copy())[None].to(device).float()
        self.right_current_depth = transform(right_current_depth.copy())[None].to(device).float()
        focal = 4161.221
        cx = 1445.577
        cy = 984.686
        camera_matrix = np.array([[focal, 0., cx],
                                  [0., focal, cy],
                                  [0., 0., 1.]])
        camera_baseline = 176
        self.cameras_calibration = CamerasCalibration(camera_baseline, camera_matrix, camera_matrix, device)
        self.lambda_s = 0.85
        self.current_position = torch.tensor([[camera_baseline, 0, 0]]).to(device).float()
        self.next_position = torch.tensor([[-camera_baseline, 0, 0]]).to(device).float()
        self.current_angle = torch.tensor([[0, 0, 0]]).to(device).float()
        self.next_angle = torch.tensor([[0, 0, 0]]).to(device).float()

    def test_transform(self):
        result = generate_relative_transformation(self.current_position, self.current_angle,
                                                  self.next_position, self.next_angle)
        self.assertEqual(result.shape, torch.Size([1, 4, 4]))
        self.assertEqual(result[0, 0, 0], 1)
        self.assertEqual(result[0, 1, 1], 1)
        self.assertEqual(result[0, 2, 2], 1)
        self.assertEqual(result[0, 3, 3], 1)

    def test_temporal_loss(self):
        loss = TemporalPhotometricConsistencyLoss(self.cameras_calibration.left_camera_matrix,
                                                  self.cameras_calibration.right_camera_matrix, self.lambda_s,
                                                  window_size=11, reduction="mean", max_val=1.0)
        output = loss.forward(self.left_current_img, self.right_current_img,
                              self.left_current_depth, self.right_current_depth,
                              self.current_position, self.current_angle,
                              self.next_position, self.next_angle)
        print(output)
        self.assertEqual(output.shape, torch.Size([]))
        self.assertFalse(torch.isnan(output))
        self.assertGreater(output, 0.05)
        self.assertLess(output, 0.11)
