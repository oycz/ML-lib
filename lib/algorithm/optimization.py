import numpy as np


def gradient_descent(X, Y, calc_gradient, calc_loss, max_iter = 100000, step_size = 0.0001, loss_threshold = 0.001):
    theta = np.zeros((X[0].size, 1))
    gradient = calc_gradient(X, Y, theta)
    theta = theta + step_size * gradient
    loss = calc_loss(X, Y, theta)
    tier = 1
    while loss > loss_threshold and tier < max_iter:
        gradient = calc_gradient(X, Y, theta)
        # print(gradient)
        theta = theta + step_size * gradient
        # print(theta)
        loss = calc_loss(X, Y, theta)
        # print(loss)
    return theta


def stochastic_gradient_descent(X, Y, theta, calc_gradient, calc_loss, max_iter = 100000, step_size = 0.0001, loss_threshold = 0.001):
    # todo
    optim_theta = np.copy(theta)
    gradient = gradient_method(X, Y, theta)

