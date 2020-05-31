import os
import sys
import unittest

import pykitti.odometry

from undeepvo.data import Downloader
from undeepvo.models import UnDeepVO
from undeepvo.problems import UnsupervisedDatasetManager, UnsupervisedDepthProblem
from undeepvo.utils import OptimizerManager, TrainingProcessHandler

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


class DatasetManagerMock(UnsupervisedDatasetManager):
    def get_train_batches(self, batch_size):
        batches = super(DatasetManagerMock, self).get_train_batches(batch_size)
        yield batches[0]

    def get_validation_batches(self, batch_size):
        batches = super(DatasetManagerMock, self).get_validation_batches(batch_size)
        yield batches[0]

    def get_test_batches(self, batch_size):
        batches = super(DatasetManagerMock, self).get_test_batches(batch_size)
        yield batches[0]


class TestUnsupervisedDepthProblem(unittest.TestCase):
    def test_unsupervised_depth_problem(self):
        sequence_8 = Downloader('08')
        if not os.path.exists("./dataset/poses"):
            print("Download dataset")
            sequence_8.download_sequence()
        lengths = (200, 30, 30)
        dataset = pykitti.odometry(sequence_8.main_dir, sequence_8.sequence_id, frames=range(0, 260, 1))
        dataset_manager = UnsupervisedDatasetManager(dataset, lenghts=lengths, num_workers=WORKERS_COUNT)
        model = UnDeepVO()
        optimizer_manger = OptimizerManager()
        criterion = None
        handler = TrainingProcessHandler()
        problem = UnsupervisedDepthProblem(model, criterion, optimizer_manger, dataset_manager, handler)
        problem.train(1)
