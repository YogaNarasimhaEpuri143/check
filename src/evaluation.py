import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

class Evaluation(ABC):
    """ Abstract class defining strategy for evaluation our models"""
    @abstractmethod
    def calculate_scores(self, Y_true: np.ndarray, Y_pred: np.ndarray):
        """
            Calculates the scores for the model
            Args:
                y_true: True labels
                y_pred: Predicted labels
            Returns:
                None
        """
        pass

class MSE(Evaluation):
    """
        Evaluation Strategy that uses MSE Score
    """
    def calculate_scores(self, Y_true: np.ndarray, Y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            mse = mean_squared_error(Y_true, Y_pred)
            logging.info("MSE Score: {}".format(mse))
            return mse
        except Exception as e:
            logging.error(f"Error in calculating R2 Score: {e}")
            raise e


class R2(Evaluation):
    """
        Evaluation Strategy that uses R2 Score
    """
    def calculate_scores(self, Y_true: np.ndarray, Y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(Y_true, Y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 Score: {e}")
            raise e
        

class RMSE(Evaluation):
    """
        Evaluation Strategy that uses RMSE Score
    """
    def calculate_scores(self, Y_true: np.ndarray, Y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            rmse = mean_squared_error(Y_true, Y_pred, squared=False)
            logging.info("MSE Score: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating R2 Score: {e}")
            raise e