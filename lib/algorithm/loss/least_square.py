import numpy as np


# gradient of least square loss function
def gradient(X, Y, theta):
    return np.dot(np.dot(X, theta) - Y, X)