import unittest
import torchvision
import torch
import kornia

from PIL import Image
import numpy as np

from undeepvo.criterion import SpatialLosses, TemporalPhotometricConsistencyLoss

from undeepvo.models import UnDeepVO

device = "cpu"


class TestSpatialLoss(unittest.TestCase):
    def read_imgs(self):
        left_current_img = Image.open("./loss_dataset/left_0.png")
        right_current_img = Image.open("./loss_dataset/right_0.png")
        left_next_img = Image.open("./loss_dataset/left_1.png")
        right_next_img = Image.open("./loss_dataset/right_1.png")
        return left_current_img, right_current_img, left_next_img, right_next_img

    def test_spatial_loss(self):
        model = UnDeepVO().to(device)

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

        left_rotation, left_position = model.pose(left_current_img)
        right_rotation, right_position = model.pose(right_current_img)

        # transformtation matrix between two cameras
        src_trans_dst = torch.tensor(((1, 0, 0, 0),
                                      (0, 1, 0, 0),
                                      (0, 0, 1, 0),
                                      (0, 0, 0, 1)))[None].to(device)
        src_trans_dst = src_trans_dst.float()

        left_current_img = left_current_img.float()
        right_current_depth = right_current_depth.float()
        left_camera_matrix = left_camera_matrix.float()

        generated_right_img = kornia.warp_frame_depth(image_src=left_current_img,
                                                      depth_dst=right_current_depth,
                                                      src_trans_dst=src_trans_dst,
                                                      camera_matrix=left_camera_matrix)

        gen_img = generated_right_img[0].detach().cpu().numpy()
        gen_img = np.swapaxes(gen_img, 0, 1)
        gen_img = np.swapaxes(gen_img, 1, 2)

        camera_baseline = 0.54
        focal_length = left_camera_matrix[0, 0, 0]
        transfrom_from_left_to_right = src_trans_dst
        lambda_position, lambda_angle, lambda_s = 1e-3, 1e-3, 1e-2

        left_current_img = left_current_img.float()
        left_current_depth = left_current_depth.float()
        left_camera_matrix = left_camera_matrix.float()

        right_current_img = right_current_img.float()
        right_current_depth = right_current_depth.float()
        right_camera_matrix = right_camera_matrix.float()

        left_position, right_position = left_position.float(), right_position.float()
        left_rotation, right_rotation = left_rotation.float(), right_rotation.float()

        spatial_losses = SpatialLosses(camera_baseline, focal_length,
                                       left_camera_matrix, right_camera_matrix, transfrom_from_left_to_right,
                                       lambda_position, lambda_angle, lambda_s)

        out = spatial_losses(left_current_img, right_current_img,
                             left_current_depth, right_current_depth,
                             left_position, right_position,
                             left_rotation, right_rotation
                             )
        self.assertEqual(out.shape, torch.Size([]))
        self.assertFalse(torch.isnan(out))
        self.assertTrue(out > 0)

    def test_temporal_loss(self):
        model = UnDeepVO().to(device)

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

        left_rotation, left_position = model.pose(left_current_img)
        right_rotation, right_position = model.pose(right_current_img)

        # transformtation matrix between two cameras
        src_trans_dst = torch.tensor(((1, 0, 0, 0),
                                      (0, 1, 0, 0),
                                      (0, 0, 1, 0),
                                      (0, 0, 0, 1)))[None].to(device)
        src_trans_dst = src_trans_dst.float()

        left_current_img = left_current_img.float()
        right_current_depth = right_current_depth.float()
        left_camera_matrix = left_camera_matrix.float()

        generated_right_img = kornia.warp_frame_depth(image_src=left_current_img,
                                                      depth_dst=right_current_depth,
                                                      src_trans_dst=src_trans_dst,
                                                      camera_matrix=left_camera_matrix)

        gen_img = generated_right_img[0].detach().cpu().numpy()
        gen_img = np.swapaxes(gen_img, 0, 1)
        gen_img = np.swapaxes(gen_img, 1, 2)

        camera_baseline = 0.54
        src_trans_dst[0, 3] = camera_baseline
        focal_length = left_camera_matrix[0, 0, 0]
        transform_from_left_to_right = src_trans_dst
        lambda_position, lambda_angle, lambda_s = 1e-3, 1e-3, 1e-2

        left_current_img = left_current_img.float()
        left_current_depth = left_current_depth.float()
        left_camera_matrix = left_camera_matrix.float()

        right_current_img = right_current_img.float()
        right_current_depth = right_current_depth.float()
        right_camera_matrix = right_camera_matrix.float()

        left_position, right_position = left_position.float(), right_position.float()
        left_rotation, right_rotation = left_rotation.float(), right_rotation.float()

        temporal_losses = TemporalPhotometricConsistencyLoss(left_camera_matrix, transform_from_left_to_right)

        out = temporal_losses(left_current_img, left_next_img, left_current_depth, left_next_depth)

        self.assertEqual(out.shape, torch.Size([]))
        self.assertFalse(torch.isnan(out))
        self.assertTrue(out > 0)