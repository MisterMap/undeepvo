import unittest

import numpy as np
import torch
import torchvision
from PIL import Image

from undeepvo.criterion import SpatialLosses, TemporalImageLosses
from undeepvo.models import UnDeepVO
from undeepvo.models.utils import init_weights
device = "cpu"


class TestLosses(unittest.TestCase):
    def read_imgs(self):
        left_current_img = Image.open("./loss_dataset/left_0.png")
        right_current_img = Image.open("./loss_dataset/right_0.png")
        left_next_img = Image.open("./loss_dataset/left_1.png")
        right_next_img = Image.open("./loss_dataset/right_1.png")
        return left_current_img, right_current_img, left_next_img, right_next_img

    def prepare_data_for_tests(self, model):
        left_current_img, right_current_img, left_next_img, right_next_img = self.read_imgs()  # return stereo pair for frame 0

        left_camera_matrix = np.array([[707.0912, 0., 601.8873],
                                       [0., 707.0912, 183.1104],
                                       [0., 0., 1.]])
        right_camera_matrix = np.array([[707.0912, 0., 601.8873],
                                        [0., 707.0912, 183.1104],
                                        [0., 0., 1.]])

        left_current_img = left_current_img.resize((384, 128))
        right_current_img = right_current_img.resize((384, 128))
        left_next_img = left_next_img.resize((384, 128))
        right_next_img = right_next_img.resize((384, 128))

        transform = torchvision.transforms.ToTensor()

        left_current_img = transform(left_current_img)[None]
        right_current_img = transform(right_current_img)[None]
        left_next_img = transform(left_next_img)[None]
        right_next_img = transform(right_next_img)[None]

        left_camera_matrix = transform(left_camera_matrix)
        right_camera_matrix = transform(right_camera_matrix)

        left_current_img, right_current_img = left_current_img.to(device), right_current_img.to(device)
        left_next_img, right_next_img = left_next_img.to(device), right_next_img.to(device)

        left_camera_matrix = left_camera_matrix.to(device)
        right_camera_matrix = right_camera_matrix.to(device)

        left_current_depth = model.depth(left_current_img).to(device)
        right_current_depth = model.depth(right_current_img).to(device)

        left_next_depth = model.depth(left_next_img).to(device)
        right_next_depth = model.depth(right_next_img).to(device)
        left_current_rotation, left_current_position = model.pose(left_current_img, right_current_img)
        right_current_rotation, right_current_position = model.pose(right_current_img, left_current_img)

        left_next_rotation, left_next_position = model.pose(left_current_img, right_current_img)
        right_next_rotation, right_next_position = model.pose(right_current_img, left_current_img)

        # transformtation matrix between two cameras
        src_trans_dst = torch.tensor(((1, 0, 0, 0),
                                      (0, 1, 0, 0),
                                      (0, 0, 1, 0),
                                      (0, 0, 0, 1)))[None].to(device)
        src_trans_dst = src_trans_dst.float()

        left_current_img = left_current_img.float()
        right_current_depth = right_current_depth.float()
        left_camera_matrix = left_camera_matrix.float()

        left_current_img = left_current_img.float()
        left_current_depth = left_current_depth.float()
        left_camera_matrix = left_camera_matrix.float()

        right_current_img = right_current_img.float()
        right_current_depth = right_current_depth.float()
        right_camera_matrix = right_camera_matrix.float()

        left_current_position, right_current_position = left_current_position.float(), right_current_position.float()
        left_current_rotation, right_current_rotation = left_current_rotation.float(), right_current_rotation.float()

        left_next_position, right_next_position = left_next_position.float(), right_next_position.float()
        left_next_rotation, right_next_rotation = left_next_rotation.float(), right_next_rotation.float()

        return left_current_img, right_current_img, left_next_img, right_next_img, \
               left_current_depth, right_current_depth, left_next_depth, right_next_depth, \
               left_current_rotation, left_current_position, right_current_rotation, right_current_position, \
               left_next_rotation, left_next_position, right_next_rotation, right_next_position, \
               src_trans_dst, left_camera_matrix, right_camera_matrix

    def test_spatial_loss(self):
        model = UnDeepVO().to(device)

        left_current_img, right_current_img, left_next_img, right_next_img, \
        left_current_depth, right_current_depth, left_next_depth, right_next_depth, \
        left_current_rotation, left_current_position, right_current_rotation, right_current_position, \
        left_next_rotation, left_next_position, right_next_rotation, right_next_position, \
        src_trans_dst, left_camera_matrix, right_camera_matrix = self.prepare_data_for_tests(model)

        camera_baseline = 0.54
        focal_length = left_camera_matrix[0, 0, 0]
        transfrom_from_left_to_right = src_trans_dst
        lambda_position, lambda_angle, lambda_s = 1e-3, 1e-3, 1e-2

        spatial_losses = SpatialLosses(camera_baseline, focal_length,
                                       left_camera_matrix, right_camera_matrix, transfrom_from_left_to_right,
                                       lambda_position, lambda_angle, lambda_s)

        out, *_ = spatial_losses(left_current_img, right_current_img,
                                 left_current_depth, right_current_depth,
                                 left_current_position, right_current_position,
                                 left_current_rotation, right_current_rotation
                                 )
        self.assertEqual(out.shape, torch.Size([]))
        self.assertFalse(torch.isnan(out))
        self.assertTrue(out > 0)

    def test_temporal_loss(self):
        model = UnDeepVO().to(device)

        left_current_img, right_current_img, left_next_img, right_next_img, \
        left_current_depth, right_current_depth, left_next_depth, right_next_depth, \
        left_current_rotation, left_current_position, right_current_rotation, right_current_position, \
        left_next_rotation, left_next_position, right_next_rotation, right_next_position, \
        src_trans_dst, left_camera_matrix, right_camera_matrix = self.prepare_data_for_tests(model)

        camera_baseline = 0.54
        src_trans_dst[0, 3] = camera_baseline
        focal_length = left_camera_matrix[0, 0, 0]
        transform_from_left_to_right = src_trans_dst
        lambda_position, lambda_angle, lambda_s = 1e-3, 1e-3, 1e-2

        temporal_losses = TemporalImageLosses(left_camera_matrix, right_camera_matrix)

        out, _ = temporal_losses(left_current_img, left_next_img, left_current_depth, left_next_depth,
                              right_current_img, right_next_img, right_current_depth, right_next_depth,
                              left_current_position, right_current_position, left_current_rotation,
                              right_current_rotation,
                              left_next_position, right_next_position, left_next_rotation, right_next_rotation
                              )

        self.assertEqual(out.shape, torch.Size([]))
        self.assertFalse(torch.isnan(out))
        self.assertTrue(out > 0)
