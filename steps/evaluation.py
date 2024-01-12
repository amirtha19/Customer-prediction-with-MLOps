import logging
import pandas as pd
from zenml import step
from sklearn.base import ClassifierMixin
from src.evaluation import F1Score, Accuracy, Recall, Precision
from typing import Tuple, Annotated
from zenml.client import Client
import mlflow  # Add this import statement

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Annotated[float, "accuracy"],
                                                                                                 Annotated[float, "recall"],
                                                                                                 Annotated[float, "f1_score"],
                                                                                                 Annotated[float, "precision"]]:
    prediction = model.predict(X_test)

    # Calculate accuracy
    accuracy_class = Accuracy()

    accuracy = accuracy_class.calculate_scores(y_test, prediction)
    mlflow.log_metric("accuracy", accuracy)

    # Calculate recall
    recall_class = Recall()
    recall = recall_class.calculate_scores(y_test, prediction)
    mlflow.log_metric("recall", recall)

    # Calculate precision
    precision_class = Precision()
    precision = precision_class.calculate_scores(y_test, prediction)
    mlflow.log_metric("precision", precision)

    # Calculate F1 score
    f1_class = F1Score()
    f1 = f1_class.calculate_scores(y_test, prediction)
    mlflow.log_metric("f1", f1)

    return accuracy, recall, f1, precision
