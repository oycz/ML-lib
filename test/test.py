import lib.regression.regression as lm
import numpy as np


# gradient of least square loss function
def gradient(X, Y, theta):
    # t is a n * 1 vector in which each row is inner product of corresponding row of X and theta
    t = np.dot(X, theta)
    # return gradient of every theta
    return np.dot(np.transpose(X), Y - t)


# value of loss function of least square, which is ||(Y - XA)||^2
def loss(X, Y, theta):
    delta = Y - np.dot(X, theta)
    delta_square = delta * delta
    return sum(delta_square)


def gradient_descent(X, Y, calc_gradient, calc_loss, max_iter=100000, step_size=0.0001, loss_threshold=0.001):
    theta = np.zeros((X[0].size, 1))
    gradient = calc_gradient(X, Y, theta)
    theta = theta + step_size * gradient
    loss = calc_loss(X, Y, theta)
    tier = 1
    while loss > loss_threshold and tier < max_iter:
        gradient = calc_gradient(X, Y, theta)
        theta = theta + step_size * gradient
        loss = calc_loss(X, Y, theta)
        tier = tier + 1
        print(loss)
    return theta


if __name__ == "__main__":
    # X = np.array([[0.57130574, 0.49885076, 0.05779497, 0.93275304, 0.84946876],
    #               [0.91112052, 0.42935098, 0.03985382, 0.74210519, 0.3799623],
    #               [0.78489266, 0.1025312, 0.19062929, 0.47770666, 0.16291279]])
    # Y = np.array([[0.01759061],
    #              [0.27069697],
    #              [0.87428128]])
    X = np.random.rand(10, 8)
    Y = np.random.rand(10, 1)
    lm1 = lm.LinearRegression(X, Y)

    # lm2 = gradient_descent(X, Y, gradient, loss)

    # print(lm1.theta)
    # print(lm1.predict(X))
    print(Y)
