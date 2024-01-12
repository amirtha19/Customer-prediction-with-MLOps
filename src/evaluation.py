import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass
class Accuracy(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating accuracy")
            accuracy = accuracy_score(y_true, y_pred)
        except Exception as e:
            logging.error(f"Error calculating accuracy: {str(e)}")
        return accuracy

class Recall(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating recall")
            
            # Use integer labels if applicable
            recall = recall_score(y_true, y_pred, pos_label=1)  # Use the appropriate integer label

        except Exception as e:
            logging.error(f"Error calculating recall: {str(e)}")
            recall = None
        return recall


class Precision(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating precision")
            precision = precision_score(y_true, y_pred, pos_label=1)  # Specify pos_label
        except Exception as e:
            logging.error(f"Error calculating precision: {str(e)}")
            precision = None
        return precision

class F1Score(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating F1 score")
            f1 = f1_score(y_true, y_pred, pos_label=1)  # Specify pos_label
        except Exception as e:
            logging.error(f"Error calculating F1 score: {str(e)}")
            f1 = None
        return f1

