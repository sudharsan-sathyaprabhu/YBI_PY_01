-Title of Project
Hand Written Digit Prediction - Classification Analysis

-Objective
To build a machine learning model that accurately predicts handwritten digits (0-9) using the MNIST dataset.

-Data Source
We’ll use the MNIST dataset, which is a large database of handwritten digits commonly used for training various image processing systems.

-Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



-Import Data
# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]


     
-Describe Data
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")
print(f"First 5 labels: {y[:5]}")


-Data Visualization
# Visualize some of the digits
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {y.iloc[i]}")
plt.show()


     
-Data Preprocessing
# Convert labels to integers
y = y.astype(np.int8)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


-Define Target Variable (y) and Feature Variables (X)
# Target variable
y = mnist["target"].astype(np.int8)

# Feature variables
X = mnist["data"]

-Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


-Modeling
# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)



     
-Model Evaluation
# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

-Prediction
# Predict a single digit
sample_digit = X_test[0].reshape(1, -1)
predicted_label = model.predict(sample_digit)
print(f"Predicted label: {predicted_label[0]}")


-Explaination
This code demonstrates a complete workflow for predicting handwritten digits using the MNIST dataset. It includes data loading, visualization, preprocessing, model training, evaluation, and making predictions.
The Logistic Regression model is used here for simplicity, but you can experiment with other models like SVM, Random Forest, or neural networks for potentially better performance.
