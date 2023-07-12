import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate sample data with a non-linear relationship
np.random.seed(0)
X = np.linspace(-5, 5, 100)
Y = 2 * np.sin(X) + np.random.normal(0, 0.5, 100)

# Define the non-linear function to fit
def nonlinear_func(x, a, b):
    return a * np.sin(b * x)

# Perform the non-linear regression
initial_guess = [1, 1]  # Initial parameter guess
params, _ = curve_fit(nonlinear_func, X, Y, p0=initial_guess)

# Generate predictions using the fitted parameters
Y_pred = nonlinear_func(X, *params)

# Plot the data and the fitted curve
plt.scatter(X, Y, label='Data')
plt.plot(X, Y_pred, color='red', label='Fitted Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
