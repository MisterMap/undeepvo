import datetime
import os.path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


class TrainingProcessHandler(object):
    def __init__(self, data_folder="logs", model_folder="model", enable_iteration_progress_bar=False,
                 model_save_key="loss", mlflow_handler=None):
        self._name = None
        self._epoch_count = 0
        self._iteration_count = 0
        self._current_epoch = 0
        self._current_iteration = 0
        self._log_folder = data_folder
        self._iteration_progress_bar = None
        self._enable_iteration_progress_bar = enable_iteration_progress_bar
        self._epoch_progress_bar = None
        self._writer = None
        self._train_metrics = {}
        self._model = None
        self._model_folder = model_folder
        self._run_name = ""
        self.train_history = {}
        self.validation_history = {}
        self._model_save_key = model_save_key
        self._previous_model_save_metric = None
        if not os.path.exists(self._model_folder):
            os.mkdir(self._model_folder)
        if not os.path.exists(self._log_folder):
            os.mkdir(self._log_folder)
        self._audio_configs = {}
        self._global_epoch_step = 0
        self._global_iteration_step = 0
        self._mlflow_handler = mlflow_handler
        self._artifacts = []

    def setup_handler(self, name, model):
        self._name = name
        self._run_name = name + "_" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self._writer = SummaryWriter(os.path.join(self._log_folder, self._run_name))
        self._model = model
        self._previous_model_save_metric = None
        self.train_history = {}
        self.validation_history = {}
        self._global_epoch_step = 0
        self._global_iteration_step = 0

    def start_callback(self, epoch_count, iteration_count, parameters=None):
        if parameters is None:
            parameters = {}
        self._epoch_count = epoch_count
        self._iteration_count = iteration_count
        self._current_epoch = 0
        self._current_iteration = 0
        self._epoch_progress_bar = tqdm(total=self._epoch_count)
        if self._enable_iteration_progress_bar:
            self._iteration_progress_bar = tqdm(total=self._iteration_count // self._epoch_count)
        if self._mlflow_handler is not None:
            self._mlflow_handler.start_callback(parameters)

    def epoch_callback(self, metrics, image_batches=None, figures=None, audios=None, texts=None):
        self._artifacts = []
        for key, value in metrics.items():
            self.validation_history.setdefault(key, []).append(value)
        self._write_epoch_metrics(metrics)
        if image_batches is not None:
            self._write_image_batches(image_batches)
        if figures is not None:
            self._write_figures(figures)
        if audios is not None:
            self._write_audios(audios)
        if texts is not None:
            self._write_texts(texts)
        if self._enable_iteration_progress_bar and self._epoch_count != self._current_epoch - 1:
            self._iteration_progress_bar.reset()
        self._epoch_progress_bar.update()
        self._epoch_progress_bar.set_postfix_str(self.metric_string("valid", metrics))
        if self.should_save_model(metrics) and self._model is not None:
            torch.save(self._model.state_dict(), os.path.join(self._model_folder, f"{self._run_name}_checkpoint.pth"))
        self._current_epoch += 1
        self._global_epoch_step += 1
        if self._mlflow_handler is not None:
            self._mlflow_handler.epoch_callback(metrics, self._current_epoch, self._artifacts)

    def iteration_callback(self, metrics):
        for key, value in metrics.items():
            self.train_history.setdefault(key, []).append(value)
        self._train_metrics = metrics
        self._write_iteration_metrics(metrics)
        if self._enable_iteration_progress_bar:
            self._iteration_progress_bar.set_postfix_str(self.metric_string("train", metrics))
            self._iteration_progress_bar.update()
        self._current_iteration += 1
        self._global_iteration_step += 1

    def finish_callback(self, metrics):
        print(self.metric_string("test", metrics))
        self._writer.close()
        if self._enable_iteration_progress_bar:
            self._iteration_progress_bar.close()
        self._epoch_progress_bar.close()
        if self._mlflow_handler is not None:
            self._mlflow_handler.finish_callback()

    @staticmethod
    def metric_string(prefix, metrics):
        result = ""
        for key, value in metrics.items():
            result += "{} {} = {:>3.3f}, ".format(prefix, key, value)
        return result[:-2]

    def _write_epoch_metrics(self, validation_metrics):
        for key, value in validation_metrics.items():
            self._writer.add_scalar(f"epoch/{key}", value, global_step=self._global_epoch_step)

    def _write_iteration_metrics(self, train_metrics):
        for key, value in train_metrics.items():
            self._writer.add_scalar(f"iteration/{key}", value, global_step=self._global_iteration_step)

    def should_save_model(self, metrics):
        if self._model_save_key not in metrics.keys():
            return True
        if self._previous_model_save_metric is None:
            self._previous_model_save_metric = metrics[self._model_save_key]
            return True
        if self._previous_model_save_metric > metrics[self._model_save_key]:
            self._previous_model_save_metric = metrics[self._model_save_key]
            return True
        return False

    def _write_image_batches(self, image_batches):
        for key, value in image_batches.items():
            self._writer.add_images(key, value, self._global_epoch_step, dataformats="NHWC")

    def _write_figures(self, figures):
        for key, value in figures.items():
            self._writer.add_figure(key, value, self._global_epoch_step)
            artifact_name = f"{self._log_folder}/{key}_{self._global_epoch_step:04d}.png"
            value.savefig(artifact_name)
            self._artifacts.append(artifact_name)

    def _write_audios(self, audios):
        for key, value in audios.items():
            self._writer.add_audio(key, value, self._global_epoch_step, **self._audio_configs)

    def set_audio_configs(self, configs):
        self._audio_configs = configs

    def _write_texts(self, texts):
        for key, value in texts.items():
            self._writer.add_text(key, value, self._global_epoch_step)
