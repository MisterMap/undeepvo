from .datapoint import Datapoint


class ImageSequence:
    def __init__(self, dataset):
        self._length = len(list(dataset.rgb))
        self._dataset = dataset

    def get_sequence(self, idx):
        idx = idx % (self._length - 1)  # idx = 12 % (13 - 1) = 0
        return Datapoint(self._dataset.get_rgb(idx), self._dataset.get_rgb(idx + 1))

    def get_length(self):
        return self._length
