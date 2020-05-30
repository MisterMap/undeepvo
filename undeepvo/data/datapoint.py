def merge_dicts(dict1, dict2):
    return dict2.update(dict1)


class Datapoint:
    def __init__(self, current_pair, next_pair):
        self._current_left = current_pair[0]
        self._current_right = current_pair[1]
        self._next_left = next_pair[0]
        self._next_right = next_pair[1]

    def get_current_left(self):
        return {"left_current_image": self._current_left}

    def get_current_right(self):
        return {"right_current_image": self._current_right}

    def get_next_left(self):
        return {"left_next_image": self._current_left}

    def get_next_right(self):
        return {"right_next_image": self._current_right}

    def get_left(self):
        return merge_dicts(self.get_current_left(), self.get_next_left())

    def get_right(self):
        return merge_dicts(self.get_current_right(), self.get_next_right())

    def get_current(self):
        return merge_dicts(self.get_current_left(), self.get_current_right())

    def get_next(self):
        return merge_dicts(self.get_next_left(), self.get_next_right())

    def get_data(self):
        return merge_dicts(self.get_current(), self.get_next())

    def transform(self, transform):
        self._current_left = transform(self._current_left)
        self._current_right = transform(self._current_right)
        self._next_left = transform(self._next_left)
        self._next_right = transform(self._next_right)
