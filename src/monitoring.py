import mlflow
import pandas as pd

predictions = pd.read_csv("predictions.csv")

drift = predictions["churn_prediction"].mean()

mlflow.log_metric("prediction_rate", drift)