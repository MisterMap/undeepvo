import os
import sys
import unittest

import pykitti.odometry

from undeepvo.data import Downloader
from undeepvo.models import UnDeepVO
from undeepvo.problems import UnsupervisedDatasetManager, UnsupervisedDepthProblem
from undeepvo.utils import OptimizerManager, TrainingProcessHandler
from undeepvo.criterion import UnsupervisedCriterion

if sys.platform == "win32":
    WORKERS_COUNT = 0
else:
    WORKERS_COUNT = 4


class DataLoaderMock(object):
    def __init__(self, data_loader):
        self._data_loader = data_loader

    def __len__(self):
        return 1

    def __iter__(self):
        for batch in self._data_loader:
            yield batch
            break


class DatasetManagerMock(UnsupervisedDatasetManager):
    def get_train_batches(self, batch_size):
        return DataLoaderMock(super(DatasetManagerMock, self).get_train_batches(batch_size))

    def get_validation_batches(self, batch_size):
        return DataLoaderMock(super(DatasetManagerMock, self).get_validation_batches(batch_size))

    def get_test_batches(self, batch_size):
        return DataLoaderMock(super(DatasetManagerMock, self).get_test_batches(batch_size))


class TestUnsupervisedDepthProblem(unittest.TestCase):
    def test_unsupervised_depth_problem(self):
        sequence_8 = Downloader('08')
        if not os.path.exists("./dataset/poses"):
            print("Download dataset")
            sequence_8.download_sequence()
        lengths = (200, 30, 30)
        dataset = pykitti.odometry(sequence_8.main_dir, sequence_8.sequence_id, frames=range(0, 260, 1))
        dataset_manager = DatasetManagerMock(dataset, lenghts=lengths, num_workers=WORKERS_COUNT)
        model = UnDeepVO().cuda()
        optimizer_manger = OptimizerManager()
        criterion = UnsupervisedCriterion(dataset_manager.get_cameras_calibration("cuda:0"),
                                          0.1, 1, 0.85)
        handler = TrainingProcessHandler(mlflow_tags={"name": "test"})
        problem = UnsupervisedDepthProblem(model, criterion, optimizer_manger, dataset_manager, handler,
                                           batch_size=1)
        problem.train(1)
