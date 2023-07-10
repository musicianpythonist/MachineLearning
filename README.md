In this code, we import the required libraries, generate a DataFrame with random data for age, gender, salary, and years of experience, and print the first few rows of the DataFrame using df.head().

We then assign the features (X) as the 'yearsExperience' column and the target variable (Y) as the 'salary' column from the DataFrame.

Next, we split the data into training and testing sets using train_test_split(), with 80% of the data for training and 20% for testing. We reshape the training and testing features using values.reshape(-1, 1) to ensure they are in the correct shape for the linear regression model.

We create a linear regression model, fit it to the training data, and make predictions on the test data. The mean squared error (MSE) and root mean squared error (RMSE) are then calculated to evaluate the model's performance.

Finally, we plot the actual salaries (blue dots) against the predicted salaries (red regression line) based on years of experience. The plot is labeled and displayed using plt.show().
