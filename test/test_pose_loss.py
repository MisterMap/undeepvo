import os
import sys
import unittest

import kornia
import numpy as np
import pykitti.odometry
import torch

from undeepvo.criterion import PoseLoss
from undeepvo.data import Downloader
from undeepvo.problems import UnsupervisedDatasetManager
from undeepvo.utils.math import generate_transformation, generate_pose

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


class TestPoseLoss(unittest.TestCase):
    def test_pose_loss(self):
        sequence_8 = Downloader('08')
        if not os.path.exists("./dataset/poses"):
            print("Download dataset")
            sequence_8.download_sequence()
        dataset = pykitti.odometry(sequence_8.main_dir, sequence_8.sequence_id, frames=range(0, 3, 1))
        dataset_manager = UnsupervisedDatasetManager(dataset, lenghts=(1, 1, 1), num_workers=WORKERS_COUNT)
        camera0_from_camera2_transformation = dataset.calib.T_cam0_velo.dot(np.linalg.inv(dataset.calib.T_cam2_velo))
        camera0_from_camera2_transformation = torch.from_numpy(camera0_from_camera2_transformation).cuda()[None].float()

        camera0_from_camera3_transformation = dataset.calib.T_cam0_velo.dot(np.linalg.inv(dataset.calib.T_cam3_velo))
        camera0_from_camera3_transformation = torch.from_numpy(camera0_from_camera3_transformation).cuda()[None].float()

        world_from_camera0_transformation = generate_transformation(torch.tensor([[1., 1., 1.]]),
                                                                    torch.tensor([[0.1, 0.1, 0.1]])).cuda()
        left_pose = generate_pose(kornia.compose_transformations(world_from_camera0_transformation,
                                                                 camera0_from_camera2_transformation))
        right_pose = generate_pose(kornia.compose_transformations(world_from_camera0_transformation,
                                                                  camera0_from_camera3_transformation))
        pose_loss = PoseLoss(1, 1, dataset_manager.get_cameras_calibration().transform_from_left_to_right)
        out = pose_loss(left_pose[0], right_pose[0], left_pose[1], right_pose[1])
        self.assertEqual(out.shape, torch.Size([]))
        self.assertGreaterEqual(out, 0)
        self.assertLessEqual(out, 0.001)
