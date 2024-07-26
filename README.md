# MultiClass-liver-Cirrhosis-prediction
## Introduction
This project aims to predict the status of liver cirrhosis patients (Alive, Dead, or Transplanted) based on various clinical and laboratory features. The dataset used in this project is a multi-class classification problem, where each sample belongs to one of three classes: C (Alive), CL (Alive after liver transplant), or D (Dead).

## Dataset
The dataset used in this project is available on [Kaggle](https://www.kaggle.com/competitions/playground-series-s3e26/data), which contains 300 samples with 25 features.
 It is a combination of two datasets: `train.csv` and `Cir.csv`. The `train.csv` dataset contains 300 samples with 25 features, while the `Cir.csv` dataset contains 150 samples with the same 25 features. The datasets are merged and preprocessed to handle missing values and outliers.

## Exploratory Data Analysis (EDA)
The project uses various EDA techniques to understand the dataset, including:

1. Basic statistics and plots using Pandas and Matplotlib.
2. Advanced EDA using Pandas Profiling, which provides a detailed report on the dataset, including summary statistics, correlation matrices, and distribution plots.

## Preprocessing
The preprocessing steps include:

1. Missing value imputation using the mean of each feature.
2. Outlier removal using the interquartile range (IQR) method and Isolation Forest.
3. Encoding categorical features using one-hot encoding and target encoding.
4. Scaling numerical features using standardization.

## Feature Engineering
Additional features are engineered based on the available features, including:

1. Conversion of Age from days to years.
2. Creation of new features based on whether patients are within the normal range for each feature.

## Pipelines
The project uses pipelines created in the `pipeline.py` file to preprocess and engineer features. The pipelines used are:

* `basic_pipeline`: performs basic preprocessing steps such as handling missing values and outliers, and encoding categorical features.
* `advanced_pipeline`: performs advanced preprocessing steps such as handling imbalanced data using SMOTE, removing duplicates, and encoding categorical features.
* `test_date_pipeline`: used for testing and validation.
* `beta_pipeline`: used for beta testing and refining the model.

## Modeling
The project uses XGBoost classifier as the base model, which is tuned using Bayesian optimization. The hyperparameter tuning space includes:

* n_estimators: 10 to 100
* max_depth: 5 to 50
* learning_rate: 0.01 to 0.4
* booster: gbtree or gblinear
* device: gpu

The tuned model is trained on the preprocessed dataset and evaluated using accuracy, F1-score, and log loss.

## Evaluation
The model is evaluated on the test set using the following metrics:

* Accuracy
* F1-score (macro average)
* Log loss
* Confusion matrix
* ROC-AUC curve
* Precision-recall curve

## Results
The final model achieves an accuracy of 91%, F1-score of 0.90, and log loss of 0.26 on the test set. The ROC-AUC curve and precision-recall curve demonstrate the model's performance on each class.

## Prediction
The final model is used to make predictions on the test set, and the results are saved in a CSV file.

## Requirements
To run this project, please install the required packages by running:  
`pip install -r requirements.txt`
