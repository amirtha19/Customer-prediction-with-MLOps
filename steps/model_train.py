import logging
import pandas as pd
from zenml import step
from src.model_dev import NaiveBayes
from sklearn.base import BaseEstimator, ClassifierMixin
from .config import ModelNameConfig
@step 
def train_model(X_train:pd.DataFrame,X_test:pd.DataFrame,
                y_train:pd.Series,y_test:pd.Series,
                config: ModelNameConfig) -> ClassifierMixin:
    print(X_train.shape)
    print(y_train.shape)
    model = None
    if config.model_name == "NaiveBayes":
        model = NaiveBayes()
        trained_model = model.train(X_train, y_train)
        return trained_model
    else:
        raise ValueError("Model {} not supported".format(config.model_name))
    