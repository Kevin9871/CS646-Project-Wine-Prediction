from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, Normalizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# Initialize Spark session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Load data
training_data = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/data/TrainingDataset.csv')
validation_data = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/data/ValidationDataset.csv')

# Print data to verify
print("Data loaded into Spark.")
training_data.show(5)
validation_data.show(5)

# Remove quotations from column names
for column in training_data.columns:
    training_data = training_data.withColumnRenamed(column, column.replace('"', ''))
    validation_data = validation_data.withColumnRenamed(column, column.replace('"', ''))

# Rename 'quality' column to 'label'
training_data = training_data.withColumnRenamed('quality', 'label')
validation_data = validation_data.withColumnRenamed('quality', 'label')


# Define feature columns
feature_cols = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

# Assemble feature columns into a single vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="inputFeatures")

# Normalize features
scaler = Normalizer(inputCol="inputFeatures", outputCol="features")

# Initialize Logistic Regression
lr = LogisticRegression(labelCol="label", featuresCol="features")

# Create a pipeline
pipeline = Pipeline(stages=[assembler, scaler, lr])

# Build parameter grid for CrossValidator
param_grid = ParamGridBuilder().build()

# Initialize evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

# Initialize CrossValidator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=3)

# Fit model
cv_model = crossval.fit(training_data)

# Evaluate model
f1_score = evaluator.evaluate(cv_model.transform(validation_data))
print(f"F1 Score for Our Model: {f1_score}")

# Stop Spark session
spark.stop()
