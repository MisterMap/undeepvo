import abc


# noinspection PyUnusedLocal
class DatasetManager(abc.ABC):
    def __init__(self):
        self._train_data, self._validation_data, self._test_data = [], [], []
        print(f"[Dataset] - train size = {len(self._train_data)}")
        print(f"[Dataset] - validation size = {len(self._validation_data)}")
        print(f"[Dataset] - test size = {len(self._test_data)}")

    @abc.abstractmethod
    def get_train_batches(self, batch_size):
        return []

    @abc.abstractmethod
    def get_validation_batches(self, batch_size):
        return []

    @abc.abstractmethod
    def get_test_batches(self, batch_size):
        return []

    def get_test_dataset(self):
        return self._validation_data

    def get_train_dataset(self):
        return self._train_data

    def get_validation_dataset(self):
        return self._validation_data
