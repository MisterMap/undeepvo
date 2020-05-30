from undeepvo.utils import Problem
import torch
import matplotlib.pyplot as plt


class UnsupervisedDepthProblem(Problem):
    def _train_step(self, batch):
        self._model.zero_grad()
        self._model.train()

        # Pose: tuple(rotation, translation)
        # Forward
        left_current_depth, left_current_pose = self._model(batch["left_current_image"].to(self._device))
        right_current_depth, right_current_pose = self._model(batch["right_current_image"].to(self._device))
        left_next_depth, left_next_pose = self._model(batch["left_next_image"].to(self._device))
        right_next_depth, right_next_pose = self._model(batch["right_next_image"].to(self._device))
        loss = self._criterion(left_current_depth, left_current_pose, right_current_depth, right_current_pose,
                               left_next_depth, left_next_pose, right_next_depth, right_next_pose)

        # Backward
        loss.backward()
        self._optimizer.step()
        return {"loss": loss.item()}

    def evaluate_batches(self, batches):
        self._model.eval()
        loss = 0
        with torch.no_grad():
            for x_batch, y_batch in batches:
                predictions = self._model(x_batch.to(self._device))
                loss += self._criterion(predictions, y_batch.to(self._device)).item()
        return {"loss": loss / len(batches)}

    def get_additional_data(self):
        return {"figures": self._get_depth_figure()}

    def _get_depth_figure(self):
        self._model.eval()
        image = self._dataset_manager.get_validation_dataset[0]
        with torch.no_grad():
            depth_image = self._model.depth(image[None].to(self._device))
        depth_image = depth_image[0].cpu().permute(1, 2, 0).detach().numpy()[:, :, 0]
        figure, axes = plt.subplots(2, 1, dpi=150)
        raw_image = image.cpu().permute(1, 2, 0).detach().numpy()
        axes[0].imshow(raw_image)
        axes[0].set_title("Raw image")
        axes[1].imshow(depth_image, cmap="inferno")
        axes[1].set_title("Result depth")
        return {"depth": figure}
