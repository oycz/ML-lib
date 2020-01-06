from lib.algorithm.interpolation import gen_interpolation_gradient


class Optimizer:

    def __init__(self, func, init_theta, func_gradient=None):
        self.func = func
        self.init_theta = init_theta
        if func_gradient is None:
            self.gradient = gen_interpolation_gradient(func)
        else:
            self.gradient = func_gradient


class GradientDescent(Optimizer):

    def __init__(self, func, init_theta, func_gradient, learning_rate=0.0001, max_iter=100000, loss_threshold=0.001):
        super().__init__(func, init_theta, func_gradient)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.loss_threshold = loss_threshold
        self.theta = init_theta
        self.optimize()

    def optimize(self):
        theta = self.init_theta
        grad = self.gradient(theta)
        theta = theta + self.learning_rate * grad
        func_val = self.func(theta)
        tier = 1
        while func_val > self.loss_threshold and tier < self.max_iter:
            gradient = self.gradient(theta)
            theta = theta + self.learning_rate * gradient
            func_val = self.func(theta)
            # print(theta)
            print(func_val)
            tier = tier + 1
        self.theta = theta


class SGD(Optimizer):
    # TODO
    def __init__(self):
        pass

