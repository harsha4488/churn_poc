from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("mlops_pipeline").getOrCreate()

df = spark.read.csv("data/customer_churn.csv", header=True, inferSchema=True)

# basic preprocessing
df_clean = df.dropna()

# feature selection
features = df_clean.select(
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "churn"
)

features.write.mode("overwrite").parquet("dbfs:/mlops/features/")
df.show()