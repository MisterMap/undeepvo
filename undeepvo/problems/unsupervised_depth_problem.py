import time

import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch

from undeepvo.utils import Problem
from undeepvo.utils.result_data_point import ResultDataPoint


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
        loss, spatial_photometric_loss, disparity_loss, pose_loss, temporal_loss, registration_loss \
            = self.evaluate_batch(batch)

        # Backward
        loss.backward()
        self._optimizer.step()
        end_time = time.time()
        return {"loss": loss.item(), "time": end_time - start_time,
                "spat_photo_loss": spatial_photometric_loss.item(), "disparity_loss": disparity_loss.item(),
                "pose_loss": pose_loss.item(),
                "temporal_loss": temporal_loss.item(),
                "registration_loss": registration_loss.item()}

    def evaluate_batches(self, batches):
        self._model.eval()
        total_loss, total_spatial_photometric_loss, total_disparity_loss, total_pose_loss = 0, 0, 0, 0
        total_temporal_loss = 0
        total_registration_loss = 0
        with torch.no_grad():
            for batch in batches:
                loss, spatial_photometric_loss, disparity_loss, pose_loss, temporal_loss, registration_loss \
                    = self.evaluate_batch(batch)
                total_loss += loss.item()
                total_disparity_loss += disparity_loss.item()
                total_pose_loss += pose_loss.item()
                total_spatial_photometric_loss += spatial_photometric_loss.item()
                total_temporal_loss += temporal_loss.item()
                total_registration_loss += registration_loss.item()
        return {"loss": total_loss / len(batches),
                "disparity_loss": total_disparity_loss / len(batches),
                "pose_loss": total_pose_loss / len(batches),
                "spat_photo_loss": total_spatial_photometric_loss / len(batches),
                "temporal_loss": total_temporal_loss / len(batches),
                "registration_loss": total_registration_loss / len(batches)}

    def get_additional_data(self):
        return {"figures": {**self._get_depth_figure(), **self._get_synthesized_image()}}

    def _get_depth_figure(self):
        self._model.eval()
        image = self._dataset_manager.get_validation_dataset(with_normalize=True)[0]["left_current_image"]
        with torch.no_grad():
            depth_image = self._model.depth(image[None].to(self._device))
        depth_image = depth_image[0].cpu().permute(1, 2, 0).detach().numpy()[:, :, 0]
        figure, axes = plt.subplots(2, 1, dpi=150)
        image = self._dataset_manager.get_validation_dataset(with_normalize=False)[0]["left_current_image"]
        raw_image = image.cpu().permute(1, 2, 0).detach().numpy()
        self.fill_in_axis(axes[0], raw_image, "Left current image")
        self.fill_in_axis(axes[1], depth_image, "Left current depth", depth=True)
        figure.tight_layout()
        return {"depth": figure}

    def _get_synthesized_image(self):
        self._model.eval()
        data_point = self._dataset_manager.get_validation_dataset(with_normalize=True)[0]
        with torch.no_grad():
            left_current_depth = self._model.depth(data_point["left_current_image"][None].to(self._device))
            right_current_depth = self._model.depth(data_point["right_current_image"][None].to(self._device))
        data_point = self._dataset_manager.get_validation_dataset(with_normalize=False)[0]
        left_current_image = data_point["left_current_image"][None].to(self._device)
        right_current_image = data_point["right_current_image"][None].to(self._device)
        cameras_calibration = self._dataset_manager.get_cameras_calibration(device=self._device)
        with torch.no_grad():
            generated_left_image = kornia.warp_frame_depth(right_current_image,
                                                           left_current_depth,
                                                           cameras_calibration.transform_from_left_to_right,
                                                           cameras_calibration.left_camera_matrix)
            generated_right_image = kornia.warp_frame_depth(left_current_image,
                                                            right_current_depth,
                                                            torch.inverse(
                                                                cameras_calibration.transform_from_left_to_right),
                                                            cameras_calibration.left_camera_matrix)

        figure = plt.figure(dpi=200, figsize=(9, 6))

        plt.subplot(3, 2, 1)
        image = left_current_image[0].cpu().permute(1, 2, 0).detach().numpy()
        plt.imshow(np.clip(image, 0, 1))
        self.set_title("Left current image")

        plt.subplot(3, 2, 2)
        image = right_current_image[0].cpu().permute(1, 2, 0).detach().numpy()
        plt.imshow(np.clip(image, 0, 1))
        self.set_title("Right current image")

        plt.subplot(3, 2, 3)
        depth_image = left_current_depth[0].detach().cpu().permute(1, 2, 0).numpy()[:, :, 0]
        plt.imshow(np.clip(depth_image, 0, 100) / 100, cmap="inferno")
        self.set_title("Left current depth")

        plt.subplot(3, 2, 4)
        depth_image = right_current_depth[0].detach().cpu().permute(1, 2, 0).numpy()[:, :, 0]
        plt.imshow(np.clip(depth_image, 0, 100) / 100, cmap="inferno")
        self.set_title("Right current depth")

        plt.subplot(3, 2, 5)
        image = generated_left_image[0].cpu().permute(1, 2, 0).detach().numpy()
        plt.imshow(np.clip(image, 0, 1))
        self.set_title("Generated left image")

        plt.subplot(3, 2, 6)
        image = generated_right_image[0].cpu().permute(1, 2, 0).detach().numpy()
        plt.imshow(np.clip(image, 0, 1))
        self.set_title("Generated right image")
        return {"generated": figure}

    @staticmethod
    def set_title(caption="None"):
        plt.title(caption)
        plt.xticks([])
        plt.yticks([])

    @staticmethod
    def fill_in_axis(axis, image, caption="None", depth=False):
        if not depth:
            axis.imshow(np.clip(image, 0, 1))
        else:
            axis.imshow(np.clip(image, 0, 100) / 100, cmap="inferno")
        axis.set_title(caption)
        axis.set_xticks([])
        axis.set_yticks([])
