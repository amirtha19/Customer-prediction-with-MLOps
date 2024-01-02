import numpy as np
import pandas as pd
from zenml import pipeline,step
from zenml.config import DockerSettings
from materializer.custom_materializer import cs_materializer
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import(MLFLowModelDeployer,)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters,Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
@pipeline(enable_cache=True,settings={"docker_settings": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy:float = 0.82,
    workers:int=1,
    timeout:int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df()
    X_train, X_test,y_train,y_test = clean_df(df)
    model = train_model(X_train,X_test,y_train,y_test)
    accuracy, recall, f1, precision = evaluate_model(model, X_test, y_test)
    if accuracy > min_accuracy:

