import torch
import numpy as np
from undeepvo.data.supervised import GroundTruthDataset
from undeepvo.problems import SupervisedDatasetManager


class DepthModelEvaluator:
    def __init__(self, model, batch_size=8, length=1000, lengths=(1, 1, 998)):
        self._model = model
        self._batch_size = batch_size
        self._length = length
        self._lengths = lengths

    def calculate_metrics(self):
        self._model.eval()
        metrics = {}
        depth_dataset = GroundTruthDataset(length=self._length, velodyne=False)
        s_dataset_manager = SupervisedDatasetManager(depth_dataset, lengths=self._lengths)
        batches = s_dataset_manager.get_test_batches(self._batch_size)
        for img, true in s_dataset_manager.get_test_dataset():
            pred = self.get_depth_from_image(img)
            self.append_to_metrics(metrics, self.compute_depth_errors(pred, true))
        final_metrics = {}
        for key in metrics.keys():
            final_metrics[key] = np.array(metrics[key]).mean()
        return final_metrics

    def get_depth_from_image(self, img):
        with torch.no_grad():
            return self._model.depth(img[None].to("cuda:0"))

    @staticmethod
    def convert_to_numpy(pred, true):
        return pred[0].cpu().permute(1, 2, 0).detach().numpy()[:, :, 0], true.cpu().detach().squeeze(0).numpy()

    @staticmethod
    def append_to_metrics(metrics, new_metric):
        if len(metrics.keys()) == 0:
            for key in new_metric.keys():
                metrics[key] = [new_metric[key]]
        else:
            for key in new_metric.keys():
                metrics[key].append(new_metric[key])
        return metrics

    def compute_depth_errors(self, pred, true):
        pred, true = self.convert_to_numpy(pred, true)
        return {"abs_rel": self.calc_abs_rel(pred, true),
                "sq_rel": self.calc_sq_rel(pred, true),
                "rmse": self.calc_rmse(pred, true),
                "rmse_log": self.calc_rmse_log(pred, true)}

    @staticmethod
    def calc_abs_rel(pred_depth: np.ndarray, true_depth: np.ndarray):
        idxs = true_depth > 0
        return (np.fabs(true_depth[idxs] - pred_depth[idxs]) / pred_depth[idxs]).mean()

    @staticmethod
    def calc_sq_rel(pred_depth: np.ndarray, true_depth: np.ndarray):
        idxs = true_depth > 0
        return (np.power(true_depth[idxs] - pred_depth[idxs], 2) / pred_depth[idxs]).mean()

    @staticmethod
    def calc_rmse(pred_depth: np.ndarray, true_depth: np.ndarray):
        idxs = true_depth > 0
        return np.sqrt((np.power(true_depth[idxs] - pred_depth[idxs], 2)).mean())

    @staticmethod
    def calc_rmse_log(pred_depth: np.ndarray, true_depth: np.ndarray):
        idxs = true_depth > 0
        return np.sqrt((np.power(np.log(true_depth[idxs]) - np.log(pred_depth[idxs]), 2)).mean())
