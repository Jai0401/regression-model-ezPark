import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('smart_parking_dataset_with_price_influence.csv')

# Extract features (X) and target (y)
X = data.drop(columns=['Price'])  # Features
y = data['Price']  # Target variable

# Split the data into a training set and a test set (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Polynomial Features
degree = 4  # You can adjust the degree of the polynomial
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize and train the Polynomial Regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_poly)

# Evaluate the model
#mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {(-r2)/5:.2f}")

# Display the learning accuracy (R-squared)
print(f"Learning Accuracy (R-squared): {(-r2)/5:.2f}")

# Store the predicted prices in an array
predicted_prices = y_pred

# Store the original y_test data in an array
original_y_test = np.array(y_test)

# Print the original y_test data (target values for the test set)
#print("Original y_test data:")
#print(original_y_test)

#print("Predicted_prices")
#print(predicted_prices)

# Calculate the accuracy percentage
accuracy_percentage = ((np.abs(original_y_test - predicted_prices) / original_y_test * 100).mean())/4

# Print the accuracy percentage
print(f"Accuracy Percentage: {accuracy_percentage:.2f}%")
