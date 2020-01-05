import numpy as np

def gradient_descent(X, Y, theta, gradient_method):
    optim_theta = numpy.copy(theta)
    gradient = gradient_method(X, Y, theta)


def stochastic_gradient_descent(X, Y, theta, gradient_method):
    optim_theta = numpy.copy(theta)
    gradient = gradient_method(X, Y, theta)