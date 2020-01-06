import numpy as np


# gradient of absolute loss function
def gen_gradient(X, Y):
    def func(theta):
        return np.dot(np.dot(X, theta), np.transpose(Y))
    return func


# value of loss function of least square, which is ||(Y - XA)||
def gen_loss(X, Y):
    def func(theta):
        delta = Y - np.dot(X, theta)
        return abs(delta)
    return func
