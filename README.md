# CS646-Project-Wine-Prediction

Wine Prediction Project
This repository contains code and configuration files for predicting wine quality using machine learning.

Files:
DockerFile: Contains the instructions to build the Docker image for the project.
WinePrediction.py: Script to train a machine learning model for predicting wine quality.
WineTestDataPrediction.py: Script to predict wine quality on test data using the trained model.

Requirements:
Docker
Python 3.6+
Required Python packages listed in requirements.txt (not provided in this repository)

Usage:
Building the Docker Image
To build the Docker image for this project, run the following command in the directory containing the DockerFile:
docker build -t wine-prediction .

Running the Scripts:
Training the Model
To train the wine prediction model, execute the WinePrediction.py script:
python WinePrediction.py

This script will train the model using the provided dataset and save the trained model to a file.
Predicting with the Model-
To predict wine quality on test data, execute the WineTestDataPrediction.py script:
python WineTestDataPrediction.py
This script will load the trained model and make predictions on the test data.

Contributing:
Contributions are welcome! Please open an issue or submit a pull request for any changes.

License:
This project is licensed under the MIT License. See the LICENSE file for more details.
