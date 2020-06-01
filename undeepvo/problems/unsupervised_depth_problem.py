import time

import matplotlib.pyplot as plt
import torch

from undeepvo.utils import Problem
from undeepvo.utils.result_data_point import ResultDataPoint
import numpy as np


class UnsupervisedDepthProblem(Problem):
    def evaluate_batch(self, batch):
        left_current_output = ResultDataPoint(batch["left_current_image"].to(self._device)).apply_model(self._model)
        right_current_output = ResultDataPoint(batch["right_current_image"].to(self._device)).apply_model(self._model)
        left_next_output = ResultDataPoint(batch["left_next_image"].to(self._device)).apply_model(self._model)
        right_next_output = ResultDataPoint(batch["right_next_image"].to(self._device)).apply_model(self._model)
        return self._criterion(left_current_output, right_current_output, left_next_output, right_next_output)

    def _train_step(self, batch):
        start_time = time.time()
        self._model.zero_grad()
        self._model.train()

        # Forward
        loss, spatial_photometric_loss, disparity_loss, pose_loss = self.evaluate_batch(batch)

        # Backward
        loss.backward()
        self._optimizer.step()
        end_time = time.time()
        return {"loss": loss.item(), "time": end_time - start_time,
                "spat_photo_loss": spatial_photometric_loss.item(), "disparity_loss": disparity_loss.item(),
                "pose_loss": pose_loss.item()}

    def evaluate_batches(self, batches):
        self._model.eval()
        total_loss, total_spatial_photometric_loss, total_disparity_loss, total_pose_loss = 0, 0, 0, 0
        with torch.no_grad():
            for batch in batches:
                loss, spatial_photometric_loss, disparity_loss, pose_loss = self.evaluate_batch(batch)
                total_loss += loss.item()
                total_disparity_loss += disparity_loss.item()
                total_pose_loss += pose_loss.item()
                total_spatial_photometric_loss += spatial_photometric_loss.item()
        return {"loss": total_loss / len(batches),
                "disparity_loss": total_disparity_loss / len(batches),
                "pose_loss": total_pose_loss / len(batches),
                "spat_photo_loss": total_spatial_photometric_loss / len(batches)}

    def get_additional_data(self):
        return {"figures": self._get_depth_figure()}

    def _get_depth_figure(self):
        self._model.eval()
        image = self._dataset_manager.get_validation_dataset()[0]["left_current_image"]
        with torch.no_grad():
            depth_image = self._model.depth(image[None].to(self._device))
        depth_image = depth_image[0].cpu().permute(1, 2, 0).detach().numpy()[:, :, 0]
        figure, axes = plt.subplots(2, 1, dpi=150)
        raw_image = image.cpu().permute(1, 2, 0).detach().numpy()
        axes[0].imshow(np.clip(raw_image, 0, 1))
        axes[0].set_title("Left current image")
        axes[1].imshow(np.clip(depth_image, 0, 10) / 10, cmap="inferno")
        axes[1].set_title("Left current depth")
        return {"depth": figure}
