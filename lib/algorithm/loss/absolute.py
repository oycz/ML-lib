import numpy as np


# gradient of absolute loss function
def gradient(X, Y, theta):
    return np.dot(np.dot(X, theta), np.transpose(Y))


# value of loss function of absolute function, which is ||(Y - XA)||^2
def loss(X, Y, theta):
    delta = Y - np.dot(X, theta)
    return abs(delta)

