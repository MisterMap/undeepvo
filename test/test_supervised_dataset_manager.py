import unittest

import torch

from undeepvo.data.supervised import GroundTruthDataset
import sys

from undeepvo.problems.supervised_dataset_manager import SupervisedDatasetManager

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


class TestSupervisedDatasetManager(unittest.TestCase):
    @unittest.skip("")
    def test_dataset_manager(self):
        dataset = GroundTruthDataset(length=260)
        lengths = (200, 30, 30)
        dataset_manager = SupervisedDatasetManager(dataset, lenghts=lengths, num_workers=WORKERS_COUNT)

        self.assertEqual(len(dataset_manager.get_train_dataset()), lengths[0])
        self.assertEqual(len(dataset_manager.get_validation_dataset()), lengths[1])
        self.assertEqual(len(dataset_manager.get_test_dataset()), lengths[2])
        image, depth = dataset_manager.get_validation_dataset(with_normalize=True)[0]
        self.assertEqual(image.shape, torch.Size([3, 128, 384]))
        self.assertEqual(depth.shape, torch.Size([1, 128, 384]))
        batches = dataset_manager.get_train_batches(20)
        for X, y in batches:
            self.assertEqual(X.shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(y.shape, torch.Size([20, 1, 128, 384]))
            self.assertEqual(X.dtype, torch.float32)
            self.assertEqual(y.dtype, torch.uint8)
            break

        batches = dataset_manager.get_validation_batches(20)
        for X, y in batches:
            self.assertEqual(X.shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(y.shape, torch.Size([20, 1, 128, 384]))
            self.assertEqual(X.dtype, torch.float32)
            self.assertEqual(y.dtype, torch.uint8)
            break

        batches = dataset_manager.get_test_batches(20)
        for X, y in batches:
            self.assertEqual(X.shape, torch.Size([20, 3, 128, 384]))
            self.assertEqual(y.shape, torch.Size([20, 1, 128, 384]))
            self.assertEqual(X.dtype, torch.float32)
            self.assertEqual(y.dtype, torch.uint8)
            break
