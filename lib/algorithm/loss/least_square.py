import numpy as np


# gradient of least square loss function
def gradient(X, Y, theta):
    # t is a n * 1 vector in which each row is inner product of corresponding row of X and theta
    t = np.dot(X, theta)
    return np.dot(np.transpose(X), Y - t)
