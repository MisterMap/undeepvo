import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from undeepvo.data.cameras_calibration import CamerasCalibration
from undeepvo.data.datatransform_manager import DataTransformManager
from undeepvo.data.stereo_dataset import StereoDataset
from undeepvo.utils import DatasetManager


class UnsupervisedDatasetManager(DatasetManager):
    def __init__(self, kitti_dataset, num_workers=4, lengths=(80, 10, 10), final_img_size=(128, 384),
                 transform_params={"filters": True, "normalize": False}):
        dataset = StereoDataset(dataset=kitti_dataset)
        self._kitti_dataset = kitti_dataset
        train, val, test = random_split(dataset, lengths)
        self._num_workers = num_workers
        super().__init__(train, val, test)
        self._transform = DataTransformManager(self._train_dataset.dataset.get_image_size(), final_img_size,
                                               transform_params)

    def get_train_batches(self, batch_size):
        self._train_dataset.dataset.set_transform(self._transform.get_train_transform())
        return DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=self._num_workers)

    def get_validation_batches(self, batch_size, with_normalize=False):
        self._validation_dataset.dataset.set_transform(
            self._transform.get_validation_transform(with_normalize=with_normalize))
        return DataLoader(self._validation_dataset, batch_size=batch_size, shuffle=False, num_workers=self._num_workers)

    def get_test_batches(self, batch_size, with_normalize=False):
        self._test_dataset.dataset.set_transform(self._transform.get_test_transform(with_normalize=with_normalize))
        return DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False, num_workers=self._num_workers)

    def get_validation_dataset(self, with_normalize=False):
        self._validation_dataset.dataset.set_transform(
            self._transform.get_validation_transform(with_normalize=with_normalize))
        return self._validation_dataset

    @staticmethod
    def get_cameras_calibration(device="cuda:0"):
        scale = 3.
        height = 128
        width = 384
        origin_focal = 707.0912
        original_cx = 601.8873
        original_cy = 183.1104
        original_height = 384
        original_width = 1248
        focal = origin_focal / scale
        original_delta_cx = original_cx - original_width / 2
        original_delta_cy = original_cy - original_height / 2
        cx = width / 2 + original_delta_cx / scale
        cy = height / 2 + original_delta_cy / scale
        left_camera_matrix = np.array([[focal, 0., cx],
                                       [0., focal, cy],
                                       [0., 0., 1.]])
        right_camera_matrix = np.array([[focal, 0., cx],
                                        [0., focal, cy],
                                        [0., 0., 1.]])
        camera_baseline = 0.54
        return CamerasCalibration(camera_baseline, left_camera_matrix, right_camera_matrix, device)

    def get_camera0_from_left_transformation(self, device="cuda:0"):
        camera0_from_camera2_transformation = self._kitti_dataset.calib.T_cam0_velo.dot(
            np.linalg.inv(self._kitti_dataset.calib.T_cam2_velo))
        camera0_from_camera2_transformation = torch.from_numpy(camera0_from_camera2_transformation).to(device)[
            None].float()
        return camera0_from_camera2_transformation

    def get_camera0_from_right_transformation(self, device="cuda:0"):
        camera0_from_camera3_transformation = self._kitti_dataset.calib.T_cam0_velo.dot(
            np.linalg.inv(self._kitti_dataset.calib.T_cam3_velo))
        camera0_from_camera3_transformation = torch.from_numpy(camera0_from_camera3_transformation).to(device)[
            None].float()
        return camera0_from_camera3_transformation
