import kornia
import torch
import numpy as np


class PoseDataPoint:
    def __init__(self, current_matrix: np.ndarray, next_matrix: np.ndarray):
        self._current_position_name = "current_position"
        self._current_angle_name = "current_angle"
        self._next_position_name = "next_position"
        self._next_angle_name = "next_angle"
        self._delta_position_name = "delta_position"
        self._delta_angle_name = "delta_angle"

        self._current_angle = kornia.rotation_matrix_to_angle_axis(
            torch.from_numpy(current_matrix[:3, :3].astype('float32')).reshape(1, 3, 3)).permute(1, 0).squeeze()
        self._next_angle = kornia.rotation_matrix_to_angle_axis(
            torch.from_numpy(next_matrix[:3, :3].astype('float32')).reshape(1, 3, 3)).permute(1, 0).squeeze()
        self._current_position = torch.from_numpy(current_matrix[:3, 3])
        self._next_position = torch.from_numpy(next_matrix[:3, 3])

        delta_matrix = kornia.relative_transformation(torch.from_numpy(current_matrix.astype('float32')),
                                                      torch.from_numpy(next_matrix.astype('float32')))
        rotation_matrix = delta_matrix[:3, :3].reshape(1, 3, 3).clone()
        self._delta_angle = kornia.rotation_matrix_to_angle_axis(rotation_matrix).permute(1, 0).squeeze()
        self._delta_position = delta_matrix[:3, 3]

    def get_current_position(self):
        return {self._current_position_name: self._current_position}

    def get_next_position(self):
        return {self._next_position_name: self._next_position}

    def get_delta_position(self):
        return {self._delta_position_name: self._delta_position}

    def get_current_angle(self):
        return {self._current_angle_name: self._current_angle}

    def get_next_angle(self):
        return {self._next_angle_name: self._next_angle}

    def get_delta_angle(self):
        return {self._delta_angle_name: self._delta_angle}

    def get_current_state(self):
        """
        :return: dictionary in format position, angle
        """
        return {**self.get_current_position(), **self.get_current_angle()}

    def get_next_state(self):
        """
        :return: dictionary in format position, angle
        """
        return {**self.get_next_position(), **self.get_next_angle()}

    def get_delta_state(self):
        """
        :return: dictionary in format position, angle
        """
        return {**self.get_delta_position(), **self.get_delta_angle()}

    def get_data(self):
        """
        :return: dictionaries in format position, angle: 3 for each current, next and delta
        """
        return {**self.get_current_state(), **self.get_next_state(), **self.get_delta_state()}
