import unittest
import numpy as np
from PIL import Image

from undeepvo.data.supervised import GroundTruthDataset


class TestGroundTruthDataset(unittest.TestCase):
    def test_dataset(self):
        length = 1000
        dataset = GroundTruthDataset(velodyne=True, length=length)
        self.assertEqual(dataset.get_length(), length)
        idx = 0
        self.assertIsInstance(dataset.get_image(idx), np.ndarray)
        self.assertIsInstance(dataset.get_depth(idx), np.ndarray)
        # image_size
        self.assertIsInstance(dataset.get_image_size(), tuple)
        self.assertEqual(len(dataset.get_image_size()), 2)
        self.assertTrue(dataset.get_image_size()[0] < dataset.get_image_size()[1])
        # image_params
        self.assertIsInstance(dataset.get_image(idx).shape, tuple)
        self.assertEqual(len(dataset.get_image(idx).shape), 3)
        # depth_params
        self.assertIsInstance(dataset.get_depth(idx).shape, tuple)
        self.assertEqual(len(dataset.get_depth(idx).shape), 2)
