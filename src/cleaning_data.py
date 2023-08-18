import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union
from sklearn.model_selection import train_test_split

class DataStategy(ABC):
    """
        Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessingStrategy(DataStategy):
    """
        Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
            Process Data
        """
        try:
            logging.info("Preprocessing Data")
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp"
                ],
                axis = 1
            )

            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)

            # Another Processing Strategies, to encode the data, tokenize the data
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e

class DataDivideStragtegy(DataStategy):
    """
        Stragtegy for dividing data into train and test
    """
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
            Divide the data into train & test
        """

        try:
            X = data.drop(["review_score"], axis=1)
            Y = data["review_score"]
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            return X_train, X_test, Y_train, Y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e
        
class DataCleaning(DataStategy):
    """
        Class for cleaning data with processes the data and divides into train and test
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
            Handle Data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data cleaning: {e}")
            raise e
