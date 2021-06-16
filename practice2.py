import os

# Scientific and vector computation for python
import matplotlib
import numpy as np

# Plotting library
from jedi.api.refactoring import inline
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X, y = data[:, :2], data[:, 2]

m = y.size  # number of training examples
# X = np.stack([np.ones(m), X], axis=1)

fig = pyplot.figure()  # open a new figure


# print out some data points
# print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
# print('-' * 26)
# for i in range(10):
#    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))


def featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).

    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).

    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu.
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation
    in sigma.

    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature.

    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    features = len(X[1, :])
    mean = []
    sd = []

    # =========================== YOUR CODE HERE =====================
    for f in range(features):
        mu[f] = (np.mean(X[:, f]))
        sigma[f] = (np.std(X[:, f]))
        #X_norm[:, f] = (X - mu[f])/sigma[f]
    X_norm = (X - mu) / sigma
    # ================================================================
    return X_norm, mu, sigma


# call featureNormalize on the loaded data
X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)


# ====================== YOUR CODE HERE =======================
# pyplot.plot(X, y, 'ro', ms=10, mec='k')
# pyplot.ylabel('Profit in $10,000')
# pyplot.xlabel('Population of City in 10,000s')

def hypothesis(theta, X):
    return theta[0] + theta[1] * X[:, 1]


def computeCost(X, y, theta):
    # initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0

    J = (1 / (2 * m)) * np.sum((hypothesis(theta, X) - y) ** 2)

    # ====================== YOUR CODE HERE =====================

    # ===========================================================
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]  # number of training examples

    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    theta0 = []
    theta1 = []
    J_history = []  # Use a python list to save cost in every iteration

    X_Values = X[:, 1]
    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        h = hypothesis(theta, X)
        theta[0] = theta[0] - (alpha / m) * (np.sum(h - y))
        theta[1] = theta[1] - (alpha / m) * (np.dot((h - y), X[:, 1]))
        # ===================================================================
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
        # print(J_history)
    return theta, J_history


# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

# theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
# print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
# print('Expected theta values (approximately): [-3.6303, 1.1664]')
