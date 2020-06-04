import time

import albumentations
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch

from undeepvo.data.datatransform_manager import DataTransformManager
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
        loss, spatial_photometric_loss, disparity_loss, pose_loss, temporal_loss = self.evaluate_batch(batch)

        # Backward
        loss.backward()
        self._optimizer.step()
        end_time = time.time()
        return {"loss": loss.item(), "time": end_time - start_time,
                "spat_photo_loss": spatial_photometric_loss.item(), "disparity_loss": disparity_loss.item(),
                "pose_loss": pose_loss.item(),
                "temporal_loss": temporal_loss.item()}

    def evaluate_batches(self, batches):
        self._model.eval()
        total_loss, total_spatial_photometric_loss, total_disparity_loss, total_pose_loss = 0, 0, 0, 0
        total_temporal_loss = 0
        with torch.no_grad():
            for batch in batches:
                loss, spatial_photometric_loss, disparity_loss, pose_loss, temporal_loss = self.evaluate_batch(batch)
                total_loss += loss.item()
                total_disparity_loss += disparity_loss.item()
                total_pose_loss += pose_loss.item()
                total_spatial_photometric_loss += spatial_photometric_loss.item()
                total_temporal_loss += temporal_loss.item()
        return {"loss": total_loss / len(batches),
                "disparity_loss": total_disparity_loss / len(batches),
                "pose_loss": total_pose_loss / len(batches),
                "spat_photo_loss": total_spatial_photometric_loss / len(batches),
                "temporal_loss": total_temporal_loss / len(batches)}

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
        self._fill_in_axis(axes[0], raw_image, "Left current image")
        self._fill_in_axis(axes[1], depth_image, "Left current depth", depth=True)
        figure.tight_layout()
        return {"depth": figure}

    def _get_synthesized_image(self):
        self._model.eval()
        image = self._dataset_manager.get_validation_dataset(with_normalize=True)[0]["left_current_image"]
        with torch.no_grad():
            depth_image = self._model.depth(image[None].to(self._device))
        left_current_depth = depth_image[0].clone()

        image = self._dataset_manager.get_validation_dataset(with_normalize=True)[0]["right_current_image"]
        with torch.no_grad():
            depth_image = self._model.depth(image[None].to(self._device))
        right_current_depth = depth_image[0].clone()
        left_current_img = self._dataset_manager.get_validation_dataset(with_normalize=False)[0]["left_current_image"]
        right_current_img = self._dataset_manager.get_validation_dataset(with_normalize=False)[0]["right_current_image"]
        generated_right_img = self._get_generated_image(left_current_img, right_current_depth, left=False)
        generated_left_img = self._get_generated_image(right_current_img, left_current_depth)

        figure, axes = plt.subplots(3, 2, dpi=150)

        self._fill_in_axis(axes[0, 0], left_current_img.cpu().permute(1, 2, 0).detach().numpy(), "Left current image")
        self._fill_in_axis(axes[1, 0], left_current_depth.cpu().permute(1, 2, 0).detach().numpy()[:, :, 0],
                           "Left current depth")
        self._fill_in_axis(axes[2, 0], torch.squeeze(generated_right_img).permute(1, 2, 0).detach().numpy(),
                           "Generated right image", depth=True)

        self._fill_in_axis(axes[0, 1], right_current_img.cpu().permute(1, 2, 0).detach().numpy(), "Right current image")
        self._fill_in_axis(axes[1, 1], right_current_depth.cpu().permute(1, 2, 0).detach().numpy()[:, :, 0],
                           "Right current depth", depth=True)
        self._fill_in_axis(axes[2, 1], torch.squeeze(generated_left_img).permute(1, 2, 0).detach().numpy(),
                           "Generated left image")
        figure.tight_layout()
        return {"generated": figure}

    def _get_generated_image(self, image, depth, left=True):
        cameras_calibration = self._dataset_manager.get_cameras_calibration()
        left_camera_matrix = cameras_calibration.left_camera_matrix.to(self._device)[None].float()
        from_left_to_right_transform = cameras_calibration.transform_from_left_to_right.to(self._device)[None].float()
        from_right_to_left_transform = torch.inverse(from_left_to_right_transform)
        if left:
            return kornia.warp_frame_depth(image_src=torch.unsqueeze(image.cpu(), 0),
                                           depth_dst=torch.unsqueeze(depth.cpu(), 0),
                                           src_trans_dst=torch.squeeze(from_left_to_right_transform.cpu(),
                                                                       0),
                                           camera_matrix=torch.squeeze(left_camera_matrix.cpu(), 0),
                                           normalize_points=False)
        else:
            return kornia.warp_frame_depth(image_src=torch.unsqueeze(image.cpu(), 0),
                                           depth_dst=torch.unsqueeze(depth.cpu(), 0),
                                           src_trans_dst=torch.squeeze(from_right_to_left_transform.cpu(),
                                                                       0),
                                           camera_matrix=torch.squeeze(left_camera_matrix.cpu(), 0),
                                           normalize_points=False)

    @staticmethod
    def _fill_in_axis(axis, image, caption="None", depth=False):
        if not depth:
            axis.imshow(np.clip(image, 0, 1))
        else:
            axis.imshow(np.clip(image, 0, 100) / 100, cmap="inferno")
        axis.set_title(caption)
        axis.set_xticks([])
        axis.set_yticks([])
