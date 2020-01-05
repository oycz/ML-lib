import numpy as np
import lib.algorithm.loss.least_square as loss
import lib.algorithm.optimization as optimization
from lib.algorithm.regression import Regression


class LinearRegression(Regression):
    def __init__(self, X, Y):
        super().__init__(X, Y, linear_training, linear_predict)


def linear_training(X, Y):
    return optimization.gradient_descent(X, Y, loss.gradient, loss.loss)


def linear_predict(data, theta):
    return np.dot(data, theta)

