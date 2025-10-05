# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Data preparation: The program first loads the California Housing dataset, selecting the first three features as the input (\(X\)) and constructing a multi-output target (\(Y\)) by combining the original housing price with the seventh feature [0]. It then splits the data into training and testing sets and scales both the features and the targets to standardize their ranges [0].
2. Model training: It initializes an SGDRegressor for multi-output regression and trains the model on the scaled training data. The MultiOutputRegressor wrapper allows a single-output regressor to be used for multiple target variables simultaneously [0].
3. Prediction and evaluation: The trained model is used to make predictions on the scaled test set. These predictions are then inverse-transformed to return them to their original scale [0]. The mean squared error is calculated to evaluate the model's performance by comparing the predicted values with the actual values [0].
4. Visualization of results: The program generates two scatter plots. One visualizes the predicted versus actual housing prices, and the other shows the predicted versus actual values for the seventh feature, providing a visual assessment of the model's accuracy [0].
5. Feature importance analysis: It extracts and plots the coefficient values from each individual SGDRegressor within the MultiOutputRegressor for each of the two outputs. This provides insight into which of the three input features have the most influence on each of the two target variables [0].
   

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: VIGNESH J
RegisterNumber:  25014705
*/

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data[:, :3]  # Using only the first 3 features
Y = np.column_stack((data.target, data.data[:, 6]))  # Target + 7th feature as multi-output

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the features and targets
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

# Create and train the multi-output SGD regressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)

# Make predictions
Y_pred = multi_output_sgd.predict(X_test)

# Inverse transform to get original scale
Y_pred_original = scaler_Y.inverse_transform(Y_pred)
Y_test_original = scaler_Y.inverse_transform(Y_test)

# Calculate MSE
mse = mean_squared_error(Y_test_original, Y_pred_original)
print("Mean Square Error:", mse)
print("\nPredictions:\n", Y_pred_original[:5])
print("\nActual values:\n", Y_test_original[:5])

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for the first output (original target)
axes[0].scatter(Y_test_original[:, 0], Y_pred_original[:, 0], alpha=0.5)
axes[0].plot([Y_test_original[:, 0].min(), Y_test_original[:, 0].max()], 
             [Y_test_original[:, 0].min(), Y_test_original[:, 0].max()], 'r--')
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title('Predictions for Housing Price')

# Plot for the second output (7th feature)
axes[1].scatter(Y_test_original[:, 1], Y_pred_original[:, 1], alpha=0.5)
axes[1].plot([Y_test_original[:, 1].min(), Y_test_original[:, 1].max()], 
             [Y_test_original[:, 1].min(), Y_test_original[:, 1].max()], 'r--')
axes[1].set_xlabel('Actual Values')
axes[1].set_ylabel('Predicted Values')
axes[1].set_title(f'Predictions for {data.feature_names[6]}')

plt.tight_layout()
plt.show()

# Feature importance for each output
feature_names = data.feature_names[:3]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# For first output
coefs1 = [estimator.coef_ for estimator in multi_output_sgd.estimators_][0]
axes[0].bar(feature_names, coefs1)
axes[0].set_title('Feature Importance for Housing Price')
axes[0].set_ylabel('Coefficient Value')

# For second output
coefs2 = [estimator.coef_ for estimator in multi_output_sgd.estimators_][1]
axes[1].bar(feature_names, coefs2)
axes[1].set_title(f'Feature Importance for {data.feature_names[6]}')
axes[1].set_ylabel('Coefficient Value')

plt.tight_layout()
plt.show()
```

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)
<img width="1036" height="724" alt="Screenshot 2025-10-05 200030" src="https://github.com/user-attachments/assets/d5ee9ca6-c3d6-4a06-b09a-004ced0cc952" />
<img width="1007" height="781" alt="Screenshot 2025-10-05 200058" src="https://github.com/user-attachments/assets/49ebe7bb-7f19-4c77-aef6-1d664ab3ee30" />




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
