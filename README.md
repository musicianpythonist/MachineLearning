* SimpleLinearRegression

Features (X) are assigned as the 'yearsExperience' column and the target variable (Y) as the 'salary' column.
Data is split into training and testing sets (80% for training) and reshaped for the linear regression model.
A linear regression model is created, fitted to the training data, and used to make predictions. Mean squared error (MSE) and root mean squared error (RMSE) are calculated. Actual vs. predicted salaries are plotted.

* MultipleLinearRegression

It reads a CSV file containing housing information into a DataFrame.
The DataFrame is preprocessed by removing the last column and modifying the column names.
The data is split into training and test sets.
A linear regression model is built and trained using the training data, and predictions are made on the test data to evaluate the model's performance.

* PolynomialRegression

This code performs polynomial regression on randomly generated data using the LinearRegression model from scikit-learn.
It generates a set of random X values, computes the corresponding Y values based on a cubic equation, and stores them in arrays.
The X values are transformed using polynomial features, and the data is split into training and testing sets.
The polynomial regression model is trained on the training data, and the results are visualized by plotting the actual data points and the regression curve.