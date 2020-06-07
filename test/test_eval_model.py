import unittest

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from undeepvo import UnDeepVO
from undeepvo.problems import DepthModelEvaluator
from undeepvo.data.supervised import DataTransformManager
from undeepvo.problems import VideoVisualizer


class TestEvalModel(unittest.TestCase):
    # def test_model_loading(self):
    #     path = "checkpoint.pth"
    #     model = UnDeepVO(resnet=True).to("cuda:0")
    #     checkpoint = torch.load(path, map_location='cpu')
    #     model.load_state_dict(checkpoint)
    #
    # def test_model_evaluator(self):
    #     path = "checkpoint.pth"
    #     model = UnDeepVO(resnet=True).to("cuda:0")
    #     checkpoint = torch.load(path, map_location='cpu')
    #     model.load_state_dict(checkpoint)
    #     evaluator = DepthModelEvaluator(model)
    #     print(evaluator.calculate_metrics())

    def test_model_video(self):
        path = "checkpoint.pth"
        model = UnDeepVO(resnet=True).to("cuda:0")
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint)

        visualiser = VideoVisualizer(model, 'test2.mp4', 'out_d.mp4', 'out_img.mp4')
        visualiser.render()


if __name__ == '__main__':
    unittest.main()
