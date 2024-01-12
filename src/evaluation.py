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
<<<<<<< HEAD
=======
            accuracy = None
>>>>>>> 657c1bf529173177957a36bef2e11ad6eca77ac2
        return accuracy

class Recall(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating recall")
<<<<<<< HEAD
            
            # Use integer labels if applicable
            recall = recall_score(y_true, y_pred, pos_label=1)  # Use the appropriate integer label

=======
            recall = recall_score(y_true, y_pred, pos_label='yes')  # Specify pos_label
>>>>>>> 657c1bf529173177957a36bef2e11ad6eca77ac2
        except Exception as e:
            logging.error(f"Error calculating recall: {str(e)}")
            recall = None
        return recall

<<<<<<< HEAD

=======
>>>>>>> 657c1bf529173177957a36bef2e11ad6eca77ac2
class Precision(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating precision")
<<<<<<< HEAD
            precision = precision_score(y_true, y_pred, pos_label=1)  # Specify pos_label
=======
            precision = precision_score(y_true, y_pred, pos_label='yes')  # Specify pos_label
>>>>>>> 657c1bf529173177957a36bef2e11ad6eca77ac2
        except Exception as e:
            logging.error(f"Error calculating precision: {str(e)}")
            precision = None
        return precision

class F1Score(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating F1 score")
<<<<<<< HEAD
            f1 = f1_score(y_true, y_pred, pos_label=1)  # Specify pos_label
=======
            f1 = f1_score(y_true, y_pred, pos_label='yes')  # Specify pos_label
>>>>>>> 657c1bf529173177957a36bef2e11ad6eca77ac2
        except Exception as e:
            logging.error(f"Error calculating F1 score: {str(e)}")
            f1 = None
        return f1

