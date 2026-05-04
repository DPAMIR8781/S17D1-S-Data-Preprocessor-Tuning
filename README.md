Tumor Classification – Preprocessor Tuning

This project focuses on building and optimizing a machine learning pipeline to classify tumors as malignant or benign, with a strong emphasis on maximizing recall (i.e., detecting as many malignant cases as possible).

Objective

In medical diagnosis, missing a malignant tumor can have serious consequences.
Therefore, the goal of this project is:

Maximize the detection rate of malignant tumors (high recall)
Minimize false negatives

Dataset
Source: Provided tumor dataset
Target variable: malignant (1 = malignant, 0 = benign)
Features: Numerical medical measurements (radius, texture, area, etc.)
 Pipeline Architecture

An end-to-end pipeline was built using scikit-learn:

Pipeline([
    ("imputer", KNNImputer()),
    ("scaler", MinMaxScaler()),
    ("model", LogisticRegression())
])

Steps:
Missing Value Handling
KNNImputer used to fill missing values based on nearest neighbors
Feature Scaling
MinMaxScaler applied to normalize all features into [0,1]
Model
LogisticRegression used for classification
 Hyperparameter Optimization

We optimized the number of neighbors in KNNImputer using GridSearchCV:

param_grid = {
    "imputer__n_neighbors": [2, 5, 10]
}
 Scoring Metric:
recall (critical for medical use case)
 Best Parameter:
n_neighbors = 5
 Model Performance
 Cross-Validation (5-Fold)
Recall Score: ~0.92

 The model correctly identifies ~92% of malignant tumors.

Prediction on New Data

The trained pipeline was used to predict a new tumor sample:

prediction = pipeline.predict(new_data)

 Result:
Prediction: Malignant
Probability: ~96.8%
Malignant olma olasılığı: %96.80

Key Takeaways
Built a fully automated ML pipeline (no manual preprocessing needed)
Focused on business-critical metric (recall) instead of accuracy
Used GridSearchCV for preprocessing optimization (not just model tuning)
Demonstrated real-world ML thinking (medical risk prioritization)

 Technologies Used
Python
Pandas
Scikit-learn
Jupyter Notebook
