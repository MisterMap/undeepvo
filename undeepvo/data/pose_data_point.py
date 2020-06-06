import kornia
import numpy as np
import torch
from undeepvo.utils.math import numpy_euler_angles_from_rotation_matrix


class PoseDataPoint:
    def __init__(self, current_matrix: np.ndarray, next_matrix: np.ndarray):
        self._current_position_name = "current_position"
        self._current_angle_name = "current_angle"
        self._next_position_name = "next_position"
        self._next_angle_name = "next_angle"
        self._delta_position_name = "delta_position"
        self._delta_angle_name = "delta_angle"
        self._inverse_delta_position_name = "inverse_delta_position"
        self._inverse_delta_angle_name = "inverse_delta_angle"
        self._current_transformation_name = "current_transformation"
        self._next_transformation_name = "next_transformation"

        self._current_transformation = torch.from_numpy(current_matrix.astype('float32'))
        self._next_transformation = torch.from_numpy(next_matrix.astype('float32'))

        current_angles = numpy_euler_angles_from_rotation_matrix(current_matrix[:3, :3]).astype('float32')
        self._current_angle = torch.from_numpy(current_angles)
        next_angles = numpy_euler_angles_from_rotation_matrix(next_matrix[:3, :3]).astype('float32')
        self._next_angle = torch.from_numpy(next_angles)
        self._current_position = torch.from_numpy(current_matrix[:3, 3]).float()
        self._next_position = torch.from_numpy(next_matrix[:3, 3]).float()

        delta_matrix = np.linalg.inv(current_matrix).dot(next_matrix)
        delta_angles = numpy_euler_angles_from_rotation_matrix(delta_matrix[:3, :3]).astype('float32')
        self._delta_angle = torch.from_numpy(delta_angles)
        self._delta_position = torch.from_numpy(delta_matrix[:3, 3]).float()

        inverse_delta_matrix = np.linalg.inv(next_matrix).dot(current_matrix)
        inverse_delta_angles = numpy_euler_angles_from_rotation_matrix(inverse_delta_matrix[:3, :3]).astype('float32')
        self._inverse_delta_angle = torch.from_numpy(inverse_delta_angles)
        self._inverse_delta_position = torch.from_numpy(inverse_delta_matrix[:3, 3]).float()

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

    def get_inverse_delta_angle(self):
        return {self._inverse_delta_angle_name: self._inverse_delta_angle}

    def get_inverse_delta_position(self):
        return {self._inverse_delta_position_name: self._inverse_delta_position}

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
        return {**self.get_delta_position(), **self.get_delta_angle(),
                **self.get_inverse_delta_angle(), **self.get_inverse_delta_position()}

    def get_current_transformation(self):
        return {self._current_transformation_name: self._current_transformation}

    def get_next_transformation(self):
        return {self._next_transformation_name: self._next_transformation}

    def get_transformation(self):
        return {**self.get_current_transformation(), **self.get_next_transformation()}

    def get_data(self):
        """
        :return: dictionaries in format position, angle: 3 for each current, next and delta
        """
        return {**self.get_current_state(), **self.get_next_state(), **self.get_delta_state(),
                **self.get_transformation()}
