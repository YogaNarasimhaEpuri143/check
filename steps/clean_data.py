import logging
import pandas as pd
from zenml import step

from src.cleaning_data import DataCleaning, DataDivideStragtegy, DataPreprocessingStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "Y_train"],
    Annotated[pd.Series, "Y_test"],
]:
    """
        Cleans the data and divides into train and test

        Args:
            df: Raw data

        Returns:
            X_train : Training data
            X_test  : Testing data
            Y_train : Training labels
            Y_test  : Testing labels
    """
    try:
        process_strategy = DataPreprocessingStrategy()   # Strategy Instance.
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data(data_cleaning)   

        divide_strategy = DataDivideStragtegy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, Y_train, Y_test = data_cleaning.handle_data(data_cleaning)
        logging.info("Data Cleaning Completed!!")
        return X_train, X_test, Y_train, Y_test
    except Exception as e:
        logging.error(f"Error in cleaning Data: {e}")
        raise e