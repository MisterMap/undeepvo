import torch


class OptimizerManager(object):
    def __init__(self, optimizer_class=None, scheduler_class=None, scheduler_config=None,
                 **optimizer_config):
        self._optimizer = None
        self._scheduler = None
        self._optimizer_config = optimizer_config
        if optimizer_class is None:
            optimizer_class = torch.optim.Adam
        self._optimizer_class = optimizer_class
        if scheduler_config is None:
            scheduler_config = {}
        self._scheduler_config = scheduler_config
        self._scheduler_class = scheduler_class

    def setup_optimizer(self, parameters):
        self._optimizer = self._optimizer_class(parameters, **self._optimizer_config)
        if self._scheduler_class is not None:
            self._scheduler = self._scheduler_class(self._optimizer, **self._scheduler_config)
        return self._optimizer

    def update(self):
        if self._scheduler is not None:
            self._scheduler.step()
