from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluate_model import evaluate_model

@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, Y_train, Y_test = clean_df(df)              # No issues, upto this line.
    model = train_model(X_train, X_test, Y_train, Y_test)
    mse, r2_score = evaluate_model(model, X_test, Y_test)
# deployment.
# Tracking of the experiments.  :-> Tweak the parameters & re-run it & check the score with the previous, compare it with several metrics.
# Experiment tracker, will be implemented over train model.
# 

# Stack :- Containarizing where Project is Running, 
# 
