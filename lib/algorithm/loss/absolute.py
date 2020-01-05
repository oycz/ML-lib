import numpy as np


# gradient of absolute loss function
def gradient(X, Y, theta):
    return np.dot(np.dot(X, theta), np.transpose(Y))