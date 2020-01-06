import numpy as np
from lib.optimization.optimizer import GradientDescent
from lib.regression.loss import least_square
from lib.regression.loss import absolute


class Regression:
    def __init__(self, X, Y, loss_name, optimizer_name, learning_rate, max_iter, loss_threshold):
        # generate optimization target
        self.gradient = None
        self.func = None
        if loss_name == "least_square":
            self.gradient = least_square.gen_gradient(X, Y)
            self.func = least_square.gen_loss(X, Y)
        elif loss_name == "absolute":
            self.gradient = absolute.gen_gradient(X, Y)
            self.func = absolute.gen_loss(X, Y)
        else:
            raise NameError("no such loss function: '{}'.".format(loss_name))

        self.optimizer = None
        if optimizer_name == "gradient_descent":
            self.__optimizer__ = GradientDescent(self.func, np.zeros((X.shape[1], 1)), self.gradient,
                                                 learning_rate, max_iter, loss_threshold)
        else:
            raise NameError("no such optimizer: '{}'.".format(optimizer_name))
        self.theta = self.__optimizer__.theta


class LinearRegression(Regression):
    def __init__(self, X, Y, loss_name="least_square", optimizer_name="gradient_descent",
                 learning_rate=0.001, max_iter=100000, loss_threshold=0.0001):
        super().__init__(X, Y, loss_name, optimizer_name, learning_rate, max_iter, loss_threshold)

    def predict(self, X):
        return np.dot(X, self.theta)


class LogisticRegression(Regression):
    def __init__(self, X, Y, loss_name="least_square", optimizer_name="gradient_descent"):
        super().__init__(X, Y, loss_name, optimizer_name)
