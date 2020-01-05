import numpy as np
import lib.algorithm.loss.least_square as loss
import lib.algorithm.optimization

from lib.regression.regression import Regression


class LinearRegreesion(Regression):
    def __init__(self):
        super.__init__()


def linear_training(X, Y, training_args):
    initial_theta = np.zeros(Y.size)
    gradient = loss.gradient(X, Y, initial_theta)


def linear_predict(data, theta, classifying_args):
    return np.dot(data, theta)