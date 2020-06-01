from torch.utils.data import Dataset

from .image_sequence import ImageSequence


class StereoDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self._sequence = ImageSequence(dataset)
        self._transform = transform

    def set_transform(self, transform):
        self._transform = transform

    def __getitem__(self, index):
        datapoint = self._sequence.get_sequence(index)
        if self._transform:
            datapoint = datapoint.from_transform(self._transform(**datapoint.get_for_transform()))

        return datapoint.get_data()

    def __len__(self):
        return self._sequence.get_length()
