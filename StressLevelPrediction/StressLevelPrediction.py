# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('StressLevelDataset.csv')
# Optional: display the first few rows to inspect the data
# display(df.head())

# Assuming df1 is not used in the subsequent code, you can remove or comment it out if not needed.
# df1 = pd.read_csv('Stress_Dataset.csv')
# display(df1.head())

# Define features (X) and target (y)
X = df.drop('stress_level', axis=1)
y = df['stress_level']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a machine learning pipeline
# The pipeline first scales the data, selects features, and then applies the RidgeClassifier model
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()), # Standardize features
    ('selector', SelectKBest(score_func=f_classif, k=10)), # Added feature selection with f_classif
    ('ridge', RidgeClassifier())  # Apply Ridge Classifier model
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the test data
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
print("--------------------------------------")

# Get predictions on the test set
y_pred = pipeline.predict(X_test)
print("Predictions: ", y_pred)
print("--------------------------------------")

# Print classification report for detailed evaluation
print(classification_report(y_test, y_pred))
