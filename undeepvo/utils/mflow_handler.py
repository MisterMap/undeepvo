import os

import mlflow
import mlflow.exceptions

DEFAULT_USER_NAME = "Mikhail.Kurenkov@skoltech.ru"
DEFAULT_PASSWORD = "ILove512DataScienceCourse!"
DEFAULT_DATABRICKS_HOST = "https://community.cloud.databricks.com"
DEFAULT_EXPERIMENT_NAME = "/undeepvo/undeepvo"


class MlFlowHandler(object):
    def __init__(self, experiment_name=DEFAULT_EXPERIMENT_NAME, user_name=DEFAULT_USER_NAME, password=DEFAULT_PASSWORD,
                 host_uri="databricks", create_databricks_credential=True, databricks_host=DEFAULT_DATABRICKS_HOST, mlflow_tags={}):
        self._user_name = DEFAULT_USER_NAME
        self._password = DEFAULT_PASSWORD
        if host_uri == "databricks" and create_databricks_credential:
            self._create_databricks_credential(user_name, password, databricks_host)
        mlflow.set_tracking_uri(host_uri)
        self._experiment_name = experiment_name
        self._mlflow_client = mlflow.tracking.MlflowClient(host_uri)
        self._enable_mlflow = True
        self._mlflow_tags = mlflow_tags

    @staticmethod
    def _create_databricks_credential(user_name, password, databricks_host):
        try:
            home_folder = os.environ["HOME"]
            with open(home_folder + "/.databrickscfg", "w") as f:
                f.write("[DEFAULT]\n")
                f.write(f"host = {databricks_host}\n")
                f.write(f"username = {user_name}\n")
                f.write(f"password = {password}\n")
        except KeyError:
            pass
        except IOError as msg:
            print(f"[WARNING][MlFlowHandler] - [CreateDataBricksCredential] {msg}")

    def start_callback(self, parameters):
        try:
            mlflow.set_tags(self._mlflow_tags)
            mlflow.set_experiment(self._experiment_name)
            mlflow.start_run()
            mlflow.log_params(parameters)
        except mlflow.exceptions.MlflowException as msg:
            self._enable_mlflow = False
            print(f"[WARNING][MlFlowHandler] - [StartCallback] {msg}")
            print(f"[WARNING][MlFlowHandler] - [StartCallback] mlflow is disabled")

    def finish_callback(self):
        if not self._enable_mlflow:
            return
        try:
            mlflow.end_run()
        except mlflow.exceptions.MlflowException as msg:
            self._enable_mlflow = False
            print(f"[WARNING][MlFlowHandler] - [FinishCallback] {msg}")
            print(f"[WARNING][MlFlowHandler] - [FinishCallback] mlflow is disabled")

    def epoch_callback(self, metrics):
        if not self._enable_mlflow:
            return
        try:
            mlflow.log_metrics(metrics)
        except mlflow.exceptions.MlflowException as msg:
            self._enable_mlflow = False
            print(f"[WARNING][MlFlowHandler] - [EpochCallback] {msg}")
            print(f"[WARNING][MlFlowHandler] - [EpochCallback] mlflow is disabled")
