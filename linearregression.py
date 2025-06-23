import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("income.csv")

# Step 1: Extract feature and target
X = df['Age'].values
y = df['Income($)'].values
# Step 2: Calculate mean
x_mean = np.mean(X)
y_mean = np.mean(y)
# Step 3: Calculate weight (slope) and bias (intercept)
numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean) ** 2)
w = numerator / denominator
b = y_mean - w * x_mean

# Step 4: Make predictions
y_pred = w * X + b

# Step 5: Plot original and predicted
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Income')
plt.plot(X, y_pred, color='red', label='Predicted Line')
plt.xlabel("Age")
plt.ylabel("Income ($)")
plt.title("Linear Regression: Age vs Income")
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Print equation
print(f"Linear Regression Equation: Income = {w:.2f} * Age + {b:.2f}")
