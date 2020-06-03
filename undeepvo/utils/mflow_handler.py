import os

import mlflow
import mlflow.exceptions

DEFAULT_USER_NAME = ""
DEFAULT_PASSWORD = ""
DEFAULT_DATABRICKS_HOST = ""
DEFAULT_HOST_URI = "http://329801-ilinvalery.tmweb.ru:5001/"
DEFAULT_EXPERIMENT_NAME = "undeepvo"
CREATE_DATABRICKS_CREDENTIALS = False

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://329801-ilinvalery.tmweb.ru:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "123"
os.environ["AWS_SECRET_ACCESS_KEY"] = "12345678"


class MlFlowHandler(object):
    def __init__(self, experiment_name=DEFAULT_EXPERIMENT_NAME, user_name=DEFAULT_USER_NAME, password=DEFAULT_PASSWORD,
                 host_uri=DEFAULT_HOST_URI, create_databricks_credential=CREATE_DATABRICKS_CREDENTIALS,
                 databricks_host=DEFAULT_DATABRICKS_HOST, mlflow_tags={}, mlflow_parameters={}):
        self._user_name = DEFAULT_USER_NAME
        self._password = DEFAULT_PASSWORD
        if host_uri == "databricks" and create_databricks_credential:
            self._create_databricks_credential(user_name, password, databricks_host)
        mlflow.set_tracking_uri(host_uri)
        self._experiment_name = experiment_name
        self._mlflow_client = mlflow.tracking.MlflowClient(host_uri)
        self._enable_mlflow = True
        self._mlflow_tags = mlflow_tags
        self._mlflow_parameters = mlflow_parameters

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
            mlflow.set_experiment(self._experiment_name)
            if mlflow.active_run() is not None:
                mlflow.end_run()
            mlflow.start_run()
            mlflow.set_tags(self._mlflow_tags)
            mlflow.log_params(parameters)
            mlflow.log_params(self._mlflow_parameters)

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

    def epoch_callback(self, metrics, current_epoch=0, artifacts=None):
        if not self._enable_mlflow:
            return
        try:
            metrics["epoch"] = current_epoch
            mlflow.log_metrics(metrics, current_epoch)
            if artifacts is not None:
                for artifact in artifacts:
                    mlflow.log_artifact(artifact)
                    os.remove(artifact)
        except mlflow.exceptions.MlflowException as msg:
            self._enable_mlflow = False
            print(f"[WARNING][MlFlowHandler] - [EpochCallback] {msg}")
            print(f"[WARNING][MlFlowHandler] - [EpochCallback] mlflow is disabled")
