import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("sales_data.csv")

# Display first 5 rows
print("First 5 Rows:")
print(data.head())

# Dataset shape
print("\nShape of the Dataset:")
print(data.shape)

# Dataset information
print("\nDataset Information:")
print(data.info())

# Missing values
print("\nMissing Values:")
print(data.isnull().sum())

# -----------------------------
# Data Preprocessing
# -----------------------------

# Remove missing values
data = data.dropna()

# Convert date into datetime
data["date"] = pd.to_datetime(data["date"])

# Convert date into numerical values
data["Year"] = data["date"].dt.year
data["Month"] = data["date"].dt.month
data["Day"] = data["date"].dt.day

# Convert product family into numbers
data["family"] = data["family"].astype("category").cat.codes

# -----------------------------
# Features and Target
# -----------------------------

X = data[["store_nbr", "family", "onpromotion", "Year", "Month", "Day"]]
y = data["sales"]

print("\nSelected Features:")
print(X.head())

print("\nTarget:")
print(y.head())

# -----------------------------
# Split the dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# -----------------------------
# Train the model
# -----------------------------

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

# -----------------------------
# Evaluation
# -----------------------------

print("\nModel Evaluation")

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.grid(True)

plt.show()