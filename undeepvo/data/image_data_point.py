import numpy as np
import torch


class ImageDataPoint:
    def __init__(self, current_pair, next_pair):
        self._current_left_name = "left_current_image"  # rename
        self._current_right_name = "right_current_image"
        self._next_left_name = "left_next_image"
        self._next_right_name = "right_next_image"

        self._current_left = current_pair[0]
        self._current_right = current_pair[1]
        self._next_left = next_pair[0]
        self._next_right = next_pair[1]

    def get_current_left(self):
        return {self._current_left_name: self._current_left}

    def get_current_right(self):
        return {self._current_right_name: self._current_right}

    def get_next_left(self):
        return {self._next_left_name: self._next_left}

    def get_next_right(self):
        return {self._next_right_name: self._next_right}

    def get_left(self):
        return {**self.get_current_left(), **self.get_next_left()}

    def get_right(self):
        return {**self.get_current_right(), **self.get_next_right()}

    def get_current(self):
        return {**self.get_current_left(), **self.get_current_right()}

    def get_next(self):
        return {**self.get_next_left(), **self.get_next_right()}

    def get_data(self):
        return {**self.get_current(), **self.get_next()}

    def get_for_transform(self):
        return {"image": np.array(self._current_left),
                "image2": np.array(self._current_right),
                "image3": np.array(self._next_left),
                "image4": np.array(self._next_right)}

    def from_transform(self, dict_datapoint):
        self._current_left = torch.from_numpy(dict_datapoint["image"]).permute(2, 0, 1)
        self._current_right = torch.from_numpy(dict_datapoint["image2"]).permute(2, 0, 1)
        self._next_left = torch.from_numpy(dict_datapoint["image3"]).permute(2, 0, 1)
        self._next_right = torch.from_numpy(dict_datapoint["image4"]).permute(2, 0, 1)
        return self
