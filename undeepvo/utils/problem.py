import torch
import abc


# noinspection PyUnusedLocal
class Problem(abc.ABC):
    def __init__(self, model: torch.nn.Module, criterion, optimizer_manager, dataset_manager, training_process_handler,
                 device="cuda:0", name="", batch_size=128):
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer_manager.setup_optimizer(self._model.parameters())
        self._optimizer_manager = optimizer_manager
        self._dataset_manager = dataset_manager
        self._training_process_handler = training_process_handler
        self._device = device
        self._name = name
        self._batch_size = batch_size
        self._training_process_handler.setup_handler(self._name, self._model)

    def train(self, n_epoch=20):
        train_batches = self._dataset_manager.get_train_batches(self._batch_size)

        self._training_process_handler.start_callback(n_epoch, len(train_batches) * n_epoch)
        for i in range(n_epoch):
            for batch in train_batches:
                train_metrics = self._train_step(batch)
                self._training_process_handler.iteration_callback(metrics=train_metrics)

            self._optimizer_manager.update()
            self._training_process_handler.epoch_callback(metrics=self.get_validation_metrics(),
                                                          **self.get_additional_data())
        self._training_process_handler.finish_callback(metrics=self.get_test_metrics())

    @abc.abstractmethod
    def _train_step(self, batch):
        return {}

    def get_validation_metrics(self):
        return self.evaluate_batches(self._dataset_manager.get_validation_batches(self._batch_size))

    def get_test_metrics(self):
        return self.evaluate_batches(self._dataset_manager.get_test_batches(self._batch_size))

    def evaluate(self):
        return self.get_test_metrics()

    @abc.abstractmethod
    def evaluate_batches(self, batches):
        return {}

    @abc.abstractmethod
    def get_additional_data(self):
        return {}
