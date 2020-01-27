import numpy as np
from lib.classifier.classifier import Classifier


class Perceptron:
    def __init__(self, X, Y, learning_rate, max_iter, loss_threshold):
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

    def predict(self, X):
        return np.sign(np.dot(X, self.theta))