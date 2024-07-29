from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Initialize Spark session
spark = SparkSession.builder \
    .appName("WineQualityPrediction") \
    .getOrCreate()

# Define file paths
training_file_path = 'TrainingDataset.csv'
validation_file_path = 'ValidationDataset.csv'

# Load the data from the local filesystem
training_data = spark.read.csv(training_file_path, header=True, inferSchema=True, sep=';')
validation_data = spark.read.csv(validation_file_path, header=True, inferSchema=True, sep=';')

# Rename 'quality' column to 'label'
training_data = training_data.withColumnRenamed('quality', 'label')
validation_data = validation_data.withColumnRenamed('quality', 'label')



# Define feature columns
feature_columns = [col for col in training_data.columns if col != 'label']

# Assemble feature columns into a single vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Initialize Logistic Regression
logistic_regression = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# Create a pipeline
pipeline = Pipeline(stages=[assembler, logistic_regression])

# Train the model
model = pipeline.fit(training_data)

# Evaluate the model
predictions = model.transform(validation_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)

print(f"F1 Score: {f1_score}")

# Save the model locally
model_path = "/tmp/Model"
model.write().overwrite().save(model_path)

# Stop Spark session
spark.stop()
