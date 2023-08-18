import logging
from typing import Tuple

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

from src.evaluation import MSE, R2, RMSE

experiment_tracker = Client().active_stack.experiment_tracker

# @step
@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
        model: RegressorMixin,
        X_test: pd.DataFrame,
        Y_test: pd.Series
        ) -> Tuple[
            Annotated[float, "mse"],
            Annotated[float, "r2"]
            # Annotated[float, "rmse"]
        ]:
    """
        Evaluates model on the ingested data.

        Args:
            df: the ingested data
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(Y_test, prediction)
        mlflow.log_metric("mse", mse)

        # r2_class = R2()
        # r2 = r2_class.calculate_scores(Y_test, prediction)
        # mlflow.log_metric("r2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(Y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return mse, rmse
    except Exception as e:
        logging.error(f"Error in Evaluating Model: {e}")
        raise e
