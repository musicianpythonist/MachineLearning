import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read the CSV file into a DataFrame
df = pd.read_csv('USA_Housing.csv')

# Remove the last column from the DataFrame
df = df.iloc[:, :-1]

# Remove 'Avg. Area' from column names
df.columns = df.columns.str.replace('Avg. Area ', '')

X = df[['House Age', 'Number of Rooms', 'Number of Bedrooms', 'Income']]
Y = df['Price']

# Create a mask for randomly splitting the data into training and test sets
mask = np.random.rand(len(df)) < 0.8  # 80% of the data will be used for training
train = df[mask]  # Training set
test = df[~mask]  # Test set

# Create a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(train[['House Age', 'Number of Rooms', 'Number of Bedrooms', 'Income']], train['Price'])

# Predict the house prices for the test data
Y_pred = model.predict(test[['House Age', 'Number of Rooms', 'Number of Bedrooms', 'Income']])

# Calculate the mean squared error
mse = mean_squared_error(test['Price'], Y_pred)
print('Mean Squared Error:', mse)

# Print the coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Calculate the variance score (R^2 score)
variance_score = r2_score(test['Price'], Y_pred)
print('Variance Score (R^2 Score):', variance_score)
