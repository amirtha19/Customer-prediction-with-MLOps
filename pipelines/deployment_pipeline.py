import numpy as np
import pandas as pd
<<<<<<< HEAD
from sklearn.impute import KNNImputer
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import os,json
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters,Output
from .utils import get_data_for_test
=======
from zenml import pipeline,step
from zenml.config import DockerSettings
from materializer.custom_materializer import cs_materializer
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import(MLFLowModelDeployer,)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters,Output
>>>>>>> 657c1bf529173177957a36bef2e11ad6eca77ac2

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

<<<<<<< HEAD
requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy = 0.82

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True

@step
def deployment_trigger(accuracy:float,config:DeploymentTriggerConfig,)->bool:
    return accuracy >config.min_accuracy


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]




@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = data.drop(columns=["Sno", "Customer_number",'balance'], errors='ignore')

    # Handle missing values and standardize 'contact' column
    data['contact'] = data['contact'].replace({'Mobile': 'cellular', 'Tel': 'telephone', '?': 'unknown'})
    data['contact'].fillna(data['contact'].mode()[0], inplace=True)

    # Handle missing values and standardize 'poutcome' column
    data['poutcome'] = data['poutcome'].replace({'?': 'unknown', '????': 'unknown', 'pending': 'unknown'})
    data['poutcome'].fillna(data['poutcome'].mode()[0], inplace=True)

    # Standardize 'job' column
    data['job'] = data['job'].replace({'blue collar': 'blue-collar', '????': 'unknown'})
    data['job'].fillna(data['job'].mode()[0], inplace=True)

    # Handle missing values in 'marital' column
    data['marital'].fillna(data['marital'].mode()[0], inplace=True)

    # Standardize and handle missing values in 'education' column
    data['education'] = data['education'].str.lower().str.strip().replace({'pri mary': 'primary', 'ter tiary': 'tertiary'})
    data['education'].replace('unknown', data['education'].mode()[0], inplace=True)
    data['education'].fillna(data['education'].mode()[0], inplace=True)



    # Handle missing values in 'duration' column using KNN imputer
    data['duration'] = data['duration'].abs()
    imputer = KNNImputer(n_neighbors=5)
    data['duration'] = imputer.fit_transform(data['duration'].values.reshape(-1, 1))
    data['duration'] = data['duration'].round().astype(int)

    # Handle outliers in 'duration' column
    data['duration'] = np.clip(data['duration'], data['duration'].quantile(0.25), data['duration'].quantile(0.75))

    # Ensure 'Count_Txn' is non-negative and fill missing values with median
    data['Count_Txn'] = data['Count_Txn'].abs()
    data['Count_Txn'].fillna(data['Count_Txn'].median(), inplace=True)
    minmax_scaler = MinMaxScaler()
    numerical_features = ['campaign','age']
    data[numerical_features] = minmax_scaler.fit_transform(data[numerical_features])
    categorical_features = ['job', 'marital', 'education', 'poutcome','contact','last_contact_day']
    label_encoder = LabelEncoder()
    for feature in categorical_features:
        data[feature] = label_encoder.fit_transform(data[feature])
    
    data = pd.get_dummies(data, columns=['housing', 'loan','Gender','Insurance'], drop_first=True)

    json_list = json.loads(json.dumps(list(data.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction



@pipeline(enable_cache=True,settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path:str="data\data.xlsx",
=======
class DeploymentTriggerConfig(BaseParameters):
@pipeline(enable_cache=True,settings={"docker_settings": docker_settings})
def continuous_deployment_pipeline(
>>>>>>> 657c1bf529173177957a36bef2e11ad6eca77ac2
    min_accuracy:float = 0.82,
    workers:int=1,
    timeout:int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
<<<<<<< HEAD
    df = ingest_df(data_path="data/data.xlsx")
    X_train, X_test,y_train,y_test = clean_df(df)
    model = train_model(X_train,X_test,y_train,y_test)
    accuracy, recall, f1, precision = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(accuracy)
    mlflow_model_deployer_step(model=model,deploy_decision=deployment_decision,workers=workers,timeout=timeout,)

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)
=======
    df = ingest_df()
    X_train, X_test,y_train,y_test = clean_df(df)
    model = train_model(X_train,X_test,y_train,y_test)
    accuracy, recall, f1, precision = evaluate_model(model, X_test, y_test)
    if accuracy > min_accuracy:

>>>>>>> 657c1bf529173177957a36bef2e11ad6eca77ac2
