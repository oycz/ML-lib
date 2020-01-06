import numpy as np


# gradient of least square loss function
def gen_gradient(X, Y):
    def func(theta):
        # t is a n * 1 vector in which each row is inner product of corresponding row of X and theta
        t = np.dot(X, theta)
        # return gradient of every theta
        return np.dot(np.transpose(X), Y - t)
    return func


# value of loss function of least square, which is ||(Y - XA)||^2
def gen_loss(X, Y):
    def func(theta):
        delta = Y - np.dot(X, theta)
        delta_square = delta * delta
        return sum(delta_square)
    return func

