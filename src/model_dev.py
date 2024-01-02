import logging
from abc import ABC,abstractmethod
from sklearn.naive_bayes import GaussianNB


class Model(ABC):
    @abstractmethod
    def train(self,X_train,y_train):
        pass
class NaiveBayes(Model):
    def __init__(self):
        self.model = GaussianNB()
    
    def train(self, X_train, y_train):
        try:
            nb = self.model.fit(X_train, y_train)
            logging.info("Model trained")
            return nb
        except Exception as e:
            logging.error("Error in training model: {}".format (e))
            raise e
