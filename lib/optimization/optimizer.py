from lib.algorithm.interpolation import gen_interpolation_gradient
import numpy as np


class Optimizer:

    def __init__(self, func, init_theta, func_gradient=None):
        self.func = func
        self.init_theta = init_theta
        if func_gradient is None:
            self.gradient = gen_interpolation_gradient(func)
        else:
            self.gradient = func_gradient


class GradientDescent(Optimizer):

    def __init__(self, func, init_theta, func_gradient, learning_rate=0.0001, max_iter=100000, threshold=0.001):
        super().__init__(func, init_theta, func_gradient)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold
        self.theta = init_theta
        self.optimize()

    def optimize(self):
        theta = np.array(self.init_theta)
        tier = 0
        while True:
            grad = self.gradient(theta)
            theta = theta - self.learning_rate * grad
            norm = np.linalg.norm(grad)
            tier = tier + 1
            if norm <= self.threshold or tier >= self.max_iter:
                break
        self.theta = theta


class SGD(Optimizer):
    # TODO
    def __init__(self):
        pass

