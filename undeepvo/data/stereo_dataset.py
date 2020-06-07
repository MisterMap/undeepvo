import pykitti
from torch.utils.data import Dataset

from .image_sequence import ImageSequence
from .pose_sequence import PoseSequence


class StereoDataset(Dataset):
    def __init__(self, dataset: pykitti.odometry, transform=None):
        self._image_sequence = ImageSequence(dataset)
        self._pose_sequence = PoseSequence(dataset)
        self._transform = transform

    def set_transform(self, transform):
        self._transform = transform

    def get_image_size(self):
        return self._image_sequence.get_image_size()

    def __getitem__(self, index):
        image_data_point = self._image_sequence.get_sequence(index)
        pose_data_point = self._pose_sequence.get_sequence(index)
        if self._transform:
            image_data_point = image_data_point.from_transform(self._transform(**image_data_point.get_for_transform()))
        return {**image_data_point.get_data(), **pose_data_point.get_data()}

    def __len__(self):
        return self._image_sequence.get_length()
