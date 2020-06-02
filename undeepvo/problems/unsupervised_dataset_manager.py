from undeepvo.data.datatransform_manager import DataTransformManager
from undeepvo.data.stereo_dataset import StereoDataset
from undeepvo.utils import DatasetManager
from torch.utils.data import DataLoader, random_split
import numpy as np
from undeepvo.data.cameras_calibration import CamerasCalibration


class UnsupervisedDatasetManager(DatasetManager):
    def __init__(self, kitti_dataset, num_workers=4, lenghts=(80, 10, 10)):
        dataset = StereoDataset(dataset=kitti_dataset)
        train, val, test = random_split(dataset, lenghts)
        self._num_workers = num_workers
        super().__init__(train, val, test)

    def get_train_batches(self, batch_size):
        self._train_dataset.dataset.set_transform(DataTransformManager.get_train_transform())
        return DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=self._num_workers)

    def get_validation_batches(self, batch_size):
        self._validation_dataset.dataset.set_transform(DataTransformManager.get_validation_transform())
        return DataLoader(self._validation_dataset, batch_size=batch_size, shuffle=False, num_workers=self._num_workers)

    def get_test_batches(self, batch_size):
        self._test_dataset.dataset.set_transform(DataTransformManager.get_test_transform())
        return DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False, num_workers=self._num_workers)

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
