import unittest

import torch

from undeepvo.models import UnDeepVO


class TestUnDeepVO(unittest.TestCase):
    def test_undeepvo(self):
        model = UnDeepVO()
        input_data = torch.rand(2, 3, 384, 128)
        output = model.depth_net(input_data)
        self.assertEqual(output.shape, torch.Size([2, 1, 384, 128]))
        self.assertTrue(torch.all(output > 0))
        input_data = torch.rand(2, 3, 384, 128)
        rotation, translation = model.pose_net(input_data)
        self.assertEqual(rotation.shape, torch.Size([2, 3]))
        self.assertEqual(translation.shape, torch.Size([2, 3]))

    def test_resnet_undeepvo(self):
        model = UnDeepVO(resnet=True)
        input_data = torch.rand(2, 3, 384, 128)
        output = model.depth_net(input_data)
        self.assertEqual(output.shape, torch.Size([2, 1, 384, 128]))
        self.assertTrue(torch.all(output > 0))
        input_data = torch.rand(2, 3, 384, 128)
        rotation, translation = model.pose_net(input_data)
        self.assertEqual(rotation.shape, torch.Size([2, 3]))
        self.assertEqual(translation.shape, torch.Size([2, 3]))

    def test_inverse_sigmoid_undeepvo(self):
        model = UnDeepVO(inverse_sigmoid=True)
        input_data = torch.rand(2, 3, 384, 128)
        output = model.depth_net(input_data)
        self.assertEqual(output.shape, torch.Size([2, 1, 384, 128]))
        self.assertTrue(torch.all(output > 0))
        input_data = torch.rand(2, 3, 384, 128)
        rotation, translation = model.pose_net(input_data)
        self.assertEqual(rotation.shape, torch.Size([2, 3]))
        self.assertEqual(translation.shape, torch.Size([2, 3]))
