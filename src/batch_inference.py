import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model(
    "models:/customer_churn_model/Production"
)

data = pd.read_csv("data/new_customers.csv")

predictions = model.predict(data)

data["churn_prediction"] = predictions

data.to_csv("predictions.csv", index=False)