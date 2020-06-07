import pykitti

from .image_data_point import ImageDataPoint


class ImageSequence:
    def __init__(self, dataset: pykitti.odometry):
        self._length = len(dataset.cam2_files)
        self._dataset = dataset
        self._img_size = self._dataset.get_rgb(0)[0].size[::-1]  # PIL Image size

    def get_image_size(self):
        return self._img_size

    def get_sequence(self, idx):
        idx = idx % (self._length - 1)  # idx = 12 % (13 - 1) = 0
        return ImageDataPoint(self._dataset.get_rgb(idx), self._dataset.get_rgb(idx + 1))

    def get_length(self):
        return self._length
