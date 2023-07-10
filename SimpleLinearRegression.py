import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import random

row_count = 100

# Create a DataFrame with random data
df = pd.DataFrame()
df['age'] = [random.randint(20, 40) for _ in range(row_count)]
df['gender'] = [random.choice(['male', 'female']) for _ in range(row_count)]
df['salary'] = [random.randint(20, 70) for _ in range(row_count)]
df['yearsExperience'] = [random.randint(2, 20) for _ in range(row_count)]
print(df.head())

# Assign the features and target variables
X = df['yearsExperience']
Y = df['salary']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train.values.reshape(-1, 1), Y_train.values.reshape(-1, 1))

# Make predictions on the test data
Y_pred = model.predict(X_test.values.reshape(-1, 1))

# Calculate the mean squared error and root mean squared error
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

# Plot the actual salaries and the regression line
plt.scatter(X_test, Y_test, color='blue', label='Actual Salaries')
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression - Salary Prediction')
plt.legend()
plt.show()
