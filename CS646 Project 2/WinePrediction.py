import boto3
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Initialize Spark session
spark = SparkSession.builder \
.appName("WineQualityPrediction") \
.getOrCreate()

# Initialize boto3 client
s3 = boto3.client('s3')

# Define S3 bucket and file names
bucket_name = 'wine-set'

# Download files from S3
s3.download_file(bucket_name,'/tmp/TrainingDataset.csv')
s3.download_file(bucket_name,'/tmp/ValidationDataset.csv')

# Load the data from the local filesystem
TrainingData = spark.read.csv("/tmp/TrainingDataset.csv", header=True, inferSchema=True)
ValidationData = spark.read.csv("/tmp/ValidationDataset.csv", header=True, inferSchema=True)

# Feature transformation
featureColumns = TrainingData.columns[:-1]
assembler = VectorAssembler(inputCols=featureColumns, outputCol="features")

TrainingData = assembler.transform(TrainingData)
ValidationData = assembler.transform(ValidationData)

# Model training
LogisticReg = LogisticRegression(labelCol="quality", featuresCol="features", maxIter=10)
pipeline = Pipeline(stages=[assembler, LogisticReg])

model = pipeline.fit(TrainingData)

# Model evaluation
prediction = model.transform(ValidationData)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
accuracy = evaluator.evaluate(prediction)
print(f"F1 Score: {accuracy}")

# Save the model locally
model_path = "/tmp/Model"
model.write().overwrite().save(model_path)