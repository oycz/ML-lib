import numpy as np
import time


class Perceptron:
    def __init__(self, X, Y, lr=1, max_iteration=float("inf")):
        self.X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.Y = Y
        self.lr = lr
        self.max_iteration = max_iteration
        self.c = 0
        self.theta = np.zeros(self.X.shape[1])

    def sgd_fit(self):
        cur = 0
        while True:
            # print(self.theta)
            self.c = self.c + 1
            x = self.X[cur]
            x = np.array([x])
            y = self.Y[cur]
            y = np.array([y])
            g = self.gradient(x, y)
            self.theta = self.theta - g * self.lr
            # print(self.theta)
            # print(self.loss)
            if self.correct_num() == self.X.shape[0]:
                break
            cur = (cur + 1) % self.X.shape[0]

    def gd_fit(self):
        while True:
            # print(self.theta)
            self.c = self.c + 1
            g = self.gradient(self.X, self.Y)
            n = np.linalg.norm(g)
            if self.correct_num() == self.X.shape[0]:
                break
            self.theta = self.theta - g * self.lr

    def gradient(self, X, Y):
        grad = np.zeros(self.theta.size)
        for i in range(X.shape[0]):
            x = X[i]
            y = Y[i]
            fun_v = np.inner(x, self.theta)
            if fun_v == 0 or np.sign(fun_v) != np.sign(y):
                grad -= y * x
        return self.lr * grad

    def loss(self):
        l = 0
        for i in range(self.X.shape[0]):
            x = self.X[i]
            y = self.Y[i]
            fun_v = np.inner(x, self.theta)
            l += max(0, -y * fun_v)
        return l

    def correct_num(self):
        n = 0
        for i in range(self.X.shape[0]):
            x = self.X[i]
            y = self.Y[i]
            fun_v = np.inner(x, self.theta)
            if np.sign(fun_v) == y:
                n = n + 1
        return n


def converge_speed_tester(X, Y, method="gd"):
    def test_converge_speed(lr):
        p = Perceptron(X, Y, lr=lr)
        t1 = time.time()
        if method == "gd":
            p.gd_fit()
        else:
            p.sgd_fit()
        t2 = time.time()
        print("method: ", method, " learning rate: ", lr, " time cost: ", (t2-t1))
    return test_converge_speed


if __name__ == "__main__":
    # data processing
    data = open("perceptron.data")
    X, Y = [], []
    for line in data:
        arr_line = line.split("\n")[0].split(",")
        X_unit = arr_line[0:-1]
        X_unit = [float(n) for n in X_unit]
        Y_unit = float(arr_line[-1])

        X += [X_unit]
        Y += [Y_unit]
    X = np.array(X)
    Y = np.array(Y)

    # gradient descent fit
    p1 = Perceptron(X, Y, lr=1)
    p1.gd_fit()
    print(p1.theta)
    print(p1.c)

    # stochastic gradient descent fit
    p2 = Perceptron(X, Y, lr=1)
    p2.sgd_fit()
    print(p1.theta)
    print(p2.c)


