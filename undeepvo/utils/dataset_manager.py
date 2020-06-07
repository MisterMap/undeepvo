import abc


# noinspection PyUnusedLocal
class DatasetManager(abc.ABC):
    def __init__(self, train_dataset, validation_dataset, test_dataset):
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._test_dataset = test_dataset
        print(f"[Dataset] - train size = {len(self._train_dataset)}")
        print(f"[Dataset] - validation size = {len(self._validation_dataset)}")
        print(f"[Dataset] - test size = {len(self._test_dataset)}")

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
        return self._test_dataset

    def get_train_dataset(self):
        return self._train_dataset

    def get_validation_dataset(self):
        return self._validation_dataset
