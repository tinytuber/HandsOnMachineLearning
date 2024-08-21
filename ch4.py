#%%
#Chapter 4: Training Models
## Test normal equation with linear regression data
import numpy as np
np.random.seed(42)
m=100
X = 2 * np.random.rand(m, 1)
y = 4 + 3*X + np.random.rand(m,1)

from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# %%
## Batch Gradient Descent
eta = 0.1  # learning rate
n_epochs = 1000
m = len(X_b)  # number of instances

np.random.seed(42)
theta = np.random.randn(2, 1)  # randomly initialized model parameters

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
# %%
## Stochastic Gradient Descent
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

np.random.seed(42)
theta = np.random.randn(2, 1)  # random initialization

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)  # for SGD, do not divide by m
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients
# %%
## Sci-kit learn implementation
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())
# %%
# Mini batch Gradient Descent: useful for optimized GPUs for matrix multiplication
# %%
# Polynomial Regression can use linear model
m = 100
X = 6 * np.random.randn(m,1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m,1)

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly_features.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# %%
import matplotlib.pyplot as plt
plt.scatter(X, y)
X2 = []
y2 = []
for i in range(-20,20):
    X2.append(i)
    y2.append(lin_reg.coef_[0][1]*i**2 + lin_reg.coef_[0][0]*i + lin_reg.intercept_)

plt.plot(X2,y2)
plt.show()

# %%
