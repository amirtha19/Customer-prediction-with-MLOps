import logging
import pandas as pd
from zenml import step
from src.model_dev import NaiveBayes
from sklearn.base import BaseEstimator, ClassifierMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

# Get the experiment tracker from the active stack
active_stack = Client().active_stack
experiment_tracker = active_stack.experiment_tracker

# Check if experiment_tracker is None and set a default name
experiment_tracker_name = experiment_tracker.name if experiment_tracker else "default_experiment"

@step(experiment_tracker=experiment_tracker_name)
def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series,
                config: ModelNameConfig) -> ClassifierMixin:
    print(X_train.shape)
    print(y_train.shape)
    model = None
    if config.model_name == "NaiveBayes":
        mlflow.sklearn.autolog()
        model = NaiveBayes()
        trained_model = model.train(X_train, y_train)
        return trained_model
    else:
        raise ValueError("Model {} not supported".format(config.model_name))
