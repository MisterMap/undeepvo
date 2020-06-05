import torch
from torch.utils.data import Dataset
from .ground_truth_dataset import GroundTruthDataset
import numpy as np

class MonoDepthDataset(Dataset):
    def __init__(self, dataset: GroundTruthDataset, transforms=None):
        self._dataset = dataset
        self._transforms = None

    def set_transform(self, transform):
        self._transforms = transform

    def get_image_size(self):
        return self._dataset.get_image_size()

    def __getitem__(self, index):
        image = self._dataset.get_image(index)
        depth = self._dataset.get_depth(index)
        if self._transforms:
            to_transform = {"image": np.asarray(image), "mask": np.asarray(depth)}
            transformed = self._transforms(**to_transform)
            image = transformed["image"]
            depth = transformed["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)
        depth = depth.float()
        return image, depth

    def __len__(self):
        return self._dataset.get_length()
