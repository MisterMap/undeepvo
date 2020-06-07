import numpy as np
from tqdm.auto import tqdm
from undeepvo.utils.math import generate_transformation


class PoseEvaluator(object):
    def __init__(self, dataset_manager, model, device):
        self._dataset_manager = dataset_manager
        self._model = model
        self._device = device

    def make_trajectory(self):
        dataset = self._dataset_manager.get_validation_dataset(with_normalize=True)
        image_count = len(dataset)
        predicted_poses = [np.eye(4, dtype=np.float32)]
        for i in tqdm(range(image_count - 1)):
            data_point = dataset[i]
            current_image = data_point["right_current_image"]
            next_image = data_point["right_next_image"]
            result = self._model.pose(current_image[None].to(self._device), next_image[None].to(self._device))
            predicted_transformation = generate_transformation(result[1], result[0]).cpu().detach()[0].numpy()
            next_pose = predicted_poses[-1].dot(predicted_transformation)
            predicted_poses.append(next_pose)
        predicted_poses = np.array(predicted_poses)
        return predicted_poses
