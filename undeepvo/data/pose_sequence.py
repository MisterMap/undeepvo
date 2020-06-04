from .pose_data_point import PoseDataPoint


class PoseSequence:
    def __init__(self, dataset):
        self._length = len(dataset.poses)
        self._dataset = dataset

    def get_sequence(self, idx):
        idx = idx % (self._length - 1)  # idx = 12 % (13 - 1) = 0
        return PoseDataPoint(self._dataset.poses[idx], self._dataset.poses[idx + 1])

    def get_length(self):
        return self._length
