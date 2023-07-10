* SimpleLinearRegression

Features (X) are assigned as the 'yearsExperience' column and the target variable (Y) as the 'salary' column.
Data is split into training and testing sets (80% for training) and reshaped for the linear regression model.
A linear regression model is created, fitted to the training data, and used to make predictions. Mean squared error (MSE) and root mean squared error (RMSE) are calculated. Actual vs. predicted salaries are plotted.

* MultipleLinearRegression

It reads a CSV file containing housing information into a DataFrame.
The DataFrame is preprocessed by removing the last column and modifying the column names.
The data is split into training and test sets.
A linear regression model is built and trained using the training data, and predictions are made on the test data to evaluate the model's performance.
