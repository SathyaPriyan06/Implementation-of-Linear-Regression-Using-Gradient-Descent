# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1..Load and preprocess the dataset by importing the required libraries, reading the CSV file, applying one-hot encoding to the categorical State column, and standardizing the input features and target values.

2.Prepare the data for training by separating the independent features (X) and target variable (y), then adding a bias (intercept) column to the feature matrix.

3.Train the Linear Regression model using Batch Gradient Descent by initializing the parameters, repeatedly updating them to minimize the Mean Squared Error (MSE) cost function, and plotting the cost convergence graph.

4.Predict the profit for a new startup by preprocessing the new input in the same way as the training data, applying the learned parameters to make a prediction, and converting the predicted value back to the original scale.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: S.Sathya Priyan
RegisterNumber:  212225230255

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
file_name = "exp_3_50_Startups.csv"

possible_paths = [
    Path(file_name),
    Path.home() / "Downloads" / file_name,
    Path.home() / "Desktop" / file_name
]

file_path = next((p for p in possible_paths if p.is_file()), None)

if file_path is None:
    raise FileNotFoundError(
        f"Cannot find {file_name}.\n"
        f"Current folder: {Path.cwd()}\n"
        "Save the CSV in this folder or your Downloads folder."
    )

data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

print("First 5 rows of the dataset:")
print(data.head())

# 2. Validate required columns
required_columns = [
    "R&D Spend", "Administration",
    "Marketing Spend", "State", "Profit"
]

missing = [col for col in required_columns if col not in data.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 3. Clean data
for col in ["R&D Spend", "Administration", "Marketing Spend", "Profit"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=required_columns).reset_index(drop=True)

if len(data) < 2:
    raise ValueError("The dataset must contain at least two valid rows.")

# 4. One-hot encode the State column
X_df = pd.get_dummies(
    data.drop(columns=["Profit"]),
    columns=["State"],
    drop_first=True,
    dtype=float
)

y_raw = data[["Profit"]].to_numpy(dtype=float)

print("\nEncoded dataset:")
print(X_df.head())

print("\nShape of features:", X_df.shape)
print("Shape of target:", y_raw.shape)

# 5. Scale features and target
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X_df)
y = y_scaler.fit_transform(y_raw)

# Add bias/intercept column
m = X_scaled.shape[0]
X = np.hstack((np.ones((m, 1)), X_scaled))

print("Shape of X after adding bias:", X.shape)

# 6. Compute cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    return np.sum(errors ** 2) / (2 * m)

# 7. Batch Gradient Descent
def gradient_descent(
    X, y, learning_rate=0.01,
    num_iters=5000, tol=1e-9, verbose=False
):
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    for i in range(num_iters):
        predictions = X @ theta
        errors = predictions - y

        gradient = (X.T @ errors) / m
        theta -= learning_rate * gradient

        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if verbose and (i < 5 or (i + 1) % 500 == 0):
            print(f"Iteration {i + 1}: Cost = {cost:.8f}")

        if not np.isfinite(cost):
            raise ValueError(
                "Cost became invalid. Try a smaller learning rate."
            )

        if len(J_history) > 1:
            if abs(J_history[-2] - J_history[-1]) < tol:
                if verbose:
                    print(f"Converged at iteration {i + 1}")
                break

    return theta, J_history

# 8. Train the model
alpha = 0.01

theta, J_hist = gradient_descent(
    X, y,
    learning_rate=alpha,
    num_iters=5000,
    tol=1e-9,
    verbose=True
)

print("\nLearned Parameters (Theta):")
print(theta.flatten())

# 9. Plot cost convergence
plt.figure(figsize=(7, 4))
plt.plot(J_hist)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE / 2)")
plt.title("Cost Function Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Predict profit for a new startup
new_sample = pd.DataFrame([{
    "R&D Spend": 165349.2,
    "Administration": 136897.8,
    "Marketing Spend": 471784.1,
    "State": "New York"
}])

# Encode the new sample and align its columns
new_encoded = pd.get_dummies(
    new_sample,
    columns=["State"],
    drop_first=True,
    dtype=float
)

new_encoded = new_encoded.reindex(
    columns=X_df.columns,
    fill_value=0
)

# Scale using the SAME scaler fitted on training features
new_scaled = X_scaler.transform(new_encoded)

# Add bias column
new_design = np.hstack((
    np.ones((new_scaled.shape[0], 1)),
    new_scaled
))

# Predict and convert back to original Profit units
scaled_pred = new_design @ theta
pred_original = y_scaler.inverse_transform(scaled_pred)

print(
    "\nPredicted Profit for the new startup:",
    f"{pred_original[0, 0]:,.2f}"
)

*/

```

## Output:
<img width="997" height="902" alt="image" src="https://github.com/user-attachments/assets/0ab932c0-77d7-4fbf-9970-8de54aaacb53" />
<img width="1112" height="675" alt="image" src="https://github.com/user-attachments/assets/bd94c1ff-9242-49a7-bd27-2031316913d8" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
