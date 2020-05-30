import unittest

from torch.utils.data import DataLoader, Dataset

class DataPoint():
    def __init__(self):
        self.a = 1
        self.b = 2

class MyDataset(Dataset):
    def __getitem__(self, item):
        return DataPoint()

    def __len__(self):
        return 10


import torch
class TestUnsupervisedDatasetManager(unittest.TestCase):
    def test_ab(self):
        dataset = MyDataset()
        data_loader = DataLoader(dataset, 10)
        for batch in data_loader:
            print(batch.a)
            print(batch.b)