from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


if __name__ == "__main__":
    # Run the pipelines
    print(get_tracking_uri())
    train_pipeline(data_path="C:\\Users\\YNARASIM\\Desktop\\mlops-zenml\\customer-satisfaction\\testing\\data\\olist_customers_dataset.csv")
