import logging
import pandas as pd
from zenml import step
from src.model_dev import NaiveBayes
from sklearn.base import BaseEstimator, ClassifierMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

# Get the experiment tracker from the active stack
experiment_tracker = Client().active_stack.experiment_tracker
# Check if experiment_tracker is None and set a default name


@step(enable_cache=False,experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series,
                config: ModelNameConfig) -> ClassifierMixin:
    print(X_train.shape)
    print(y_train.shape)
    model = None
    if config.model_name == "NaiveBayes":
        mlflow.sklearn.autolog()
        model = NaiveBayes()
        
    else:
        raise ValueError("Model {} not supported".format(config.model_name))

    trained_model = model.train(X_train, y_train)
    return trained_model