import unittest
from undeepvo_utils import TrainingProcessHandler


class TestUnsupervisedDepthPrediction(unittest.TestCase):
    def test_unsupervised_depth_image(self):
        handler = TrainingProcessHandler()
        handler.setup_handler("test", None)
        handler.start_callback(1, 1)
        handler.epoch_callback({})
        handler.finish_callback({})
