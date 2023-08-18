import logging

import mlflow

import pandas as pd
from zenml import step
from zenml.client import Client

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker
# print("Here", tracker)
# active_stack = Client().active_stack
# if active_stack:
#     print(active_stack.experiment_tracker.name)
#     experiment_tracker = active_stack.experiment_tracker
#     print(experiment_tracker)
# else:
#     print("No active stack found.")

# @step
@step(experiment_tracker=experiment_tracker.name)  # Notifying this step has an experiment tracker.
def train_model(
    X_train : pd.DataFrame,
    X_test  : pd.DataFrame,
    Y_train : pd.Series,
    Y_test  : pd.Series,
    config  : ModelNameConfig
) -> RegressorMixin:
    """
        Trains model on Ingested data

        Args:
            df: Ingested data
    """
    model = None
    if config.model_name == "LinearRegression":
        logging.info("Before mlflow.sklear.autoflow command...")
        mlflow.sklearn.autolog()  # scikit-learn autologs, models, scores
        logging.info("After mlflow.sklear.autoflow command...")
        model = LinearRegressionModel()
        trained_model = model.train(X_train, Y_train)
        return trained_model
    if config.model_name == "RandomForest":
        pass
    else:
        raise ValueError("Model {} not supported".format(config.model_name))


#  MSO Deployment :- Mostly used for local deployment.
#  Celton Core    :- Deploy on AWS, GCP (Advance)
#  Continous deployment pipeline.
#  Inference Pipeline.

#  zenml disconnect
#  zenml up
#  pip install --upgrade scikit-learn | mlflow


# - zenml stack list
# - zenml stack describe
# - zenml up --blocking
# - zenml stack set <name_of_the_stack>
# - mlflow ui --backend-store-uri "<file_location>"

# ** with default stack working properly
# ** with mlflow_track, (checking ...)