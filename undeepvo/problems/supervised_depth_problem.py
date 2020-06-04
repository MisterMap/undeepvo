import time
import matplotlib.pyplot as plt
import numpy as np
import torch

from undeepvo.utils import Problem
from undeepvo.utils.result_data_point import ResultDataPoint


class SupervisedDepthProblem(Problem):
    def evaluate_batch(self, batch):
        output = self._model(batch[0].to(self._device))
        return self._criterion(output, batch[1].to(self._device))

    def _train_step(self, batch):
        start_time = time.time()
        self._model.zero_grad()
        self._model.train()

        # Forward
        loss = self.evaluate_batch(batch)
        # Backward
        loss.backward()
        self._optimizer.step()
        end_time = time.time()
        return {"loss": loss.item(), "time": end_time - start_time}

    def evaluate_batches(self, batches):
        self._model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in batches:
                loss = self.evaluate_batch(batch)
                total_loss += loss.item()
        return {"loss": total_loss / len(batches)}

    def get_additional_data(self):
        return {"figures": {**self._get_figures()}}

    def _get_figures(self):
        self._model.eval()
        image = self._dataset_manager.get_validation_dataset(with_normalize=True)[0][0]
        depth = self._dataset_manager.get_validation_dataset(with_normalize=True)[0][0]
        with torch.no_grad():
            depth_model = self._model.depth(image[None].to(self._device))
        depth_model = depth_model[0].cpu().permute(1, 2, 0).detach().numpy()[:, :, 0]
        figure, axes = plt.subplots(2, 1, dpi=150)
        image = self._dataset_manager.get_validation_dataset(with_normalize=False)[0][0]
        raw_image = image.cpu().permute(1, 2, 0).detach().numpy()
        self._fill_in_axis(axes[0], raw_image, "image")
        self._fill_in_axis(axes[1], depth, "ground_truth_depth", depth=True)
        self._fill_in_axis(axes[1], depth_model, "depth", depth=True)
        figure.tight_layout()
        return {"depth": figure}

    @staticmethod
    def _fill_in_axis(axis, image, caption="None", depth=False):
        if not depth:
            axis.imshow(np.clip(image, 0, 1))
        else:
            axis.imshow(np.clip(image, 0, 100) / 100, cmap="inferno")
        axis.set_title(caption)
        axis.set_xticks([])
        axis.set_yticks([])
