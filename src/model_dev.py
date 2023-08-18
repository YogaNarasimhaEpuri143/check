import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
        Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, Y_train, **kwargs) -> None:
        """
            Args:
                X_train : Training data
                X_test  : Testing data

            Returns: 
                None
        """
        pass

class LinearRegressionModel(Model):
    """
        Linear Regression Model
    """
    def train(self, X_train, Y_train, **kwargs):
        """
            Trains the model
            Args:
                X_train : Training data
                X_test  : Testing data

            Returns: 
                None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, Y_train)
            logging.info("Model Training Completed")
            return reg
        except Exception as e:
            logging.error(f"Error in training Model {e}")
            raise e
        
# Lot of things todo, Train the model, Validate the Assumption, Feature Engineering ...

class RandomForestModel(Model):
    def train(self, X_train, Y_train, **kwargs) -> None:
        pass