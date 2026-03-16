import mlflow

model_uri = "runs:/<run_id>/churn_model"

mlflow.register_model(
    model_uri,
    "customer_churn_model"
)