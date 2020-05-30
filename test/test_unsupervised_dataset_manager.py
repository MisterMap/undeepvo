import unittest

import torch

from undeepvo.problems import UnsupervisedDatasetManager


class TestUnsupervisedDatasetManager(unittest.TestCase):
    def test_dataset_manager(self):
        dataset_manager = UnsupervisedDatasetManager()
        self.assertEqual(len(dataset_manager.get_train_dataset()), 1000)
        self.assertEqual(len(dataset_manager.get_test_dataset()), 100)
        self.assertEqual(len(dataset_manager.get_validation_dataset()), 100)
        batches = dataset_manager.get_train_batches(20)
        for batch in batches:
            self.assertEqual(batch["left_current_image"].shape, torch.Size([20, 3, 416, 128]))
            self.assertEqual(batch["right_current_image"].shape, torch.Size([20, 3, 416, 128]))
            self.assertEqual(batch["left_next_image"].shape, torch.Size([20, 3, 416, 128]))
            self.assertEqual(batch["right_next_image"].shape, torch.Size([20, 3, 416, 128]))
            break
        batches = dataset_manager.get_validation_batches(20)
        for batch in batches:
            self.assertEqual(batch["left_current_image"].shape, torch.Size([20, 3, 416, 128]))
            self.assertEqual(batch["right_current_image"].shape, torch.Size([20, 3, 416, 128]))
            self.assertEqual(batch["left_next_image"].shape, torch.Size([20, 3, 416, 128]))
            self.assertEqual(batch["right_next_image"].shape, torch.Size([20, 3, 416, 128]))
            break
        batches = dataset_manager.get_test_batches(20)
        for batch in batches:
            self.assertEqual(batch["left_current_image"].shape, torch.Size([20, 3, 416, 128]))
            self.assertEqual(batch["right_current_image"].shape, torch.Size([20, 3, 416, 128]))
            self.assertEqual(batch["left_next_image"].shape, torch.Size([20, 3, 416, 128]))
            self.assertEqual(batch["right_next_image"].shape, torch.Size([20, 3, 416, 128]))
            break
