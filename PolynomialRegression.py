import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate random data
X = np.sort(np.random.uniform(-1, 1, 200)).reshape(-1, 1)
Yf = X**3 + 3*X**2 - X
Y = np.array([])
for y in Yf:
    Y = np.append(Y, y)

# Perform polynomial transformation
poly_transform = PolynomialFeatures(degree=2)
X_poly = poly_transform.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=42)

# Perform polynomial regression
poly_reg = LinearRegression()
poly_reg.fit(X_train, Y_train)

# Print coefficients and intercept of the regression model
print("Coefficients:", poly_reg.coef_)
print("Intercept:", poly_reg.intercept_)

# Visualizing the results
plt.scatter(X, Y, color='blue', label='Actual Data')  # Plotting the actual data points
plt.plot(X, poly_reg.predict(poly_transform.transform(X)), color='red', label='Polynomial Regression')  # Plotting the regression curve
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()
