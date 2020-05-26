import numpy as np
from collections import Counter
from lib.preprocessing.read_data import read_file


def none_penalty(lr, lmda):
    def penalty():
        return np.zeros(lr.W.shape)
    return penalty


def l1_penalty(lr, lmda):
    def penalty():
        return -lmda * np.sign(lr.W)
    return penalty


def l2_penalty(lr, lmda):
    def penalty():
        return -lmda * lr.W
    return penalty


class LogisticRegression:
    def __init__(self, X, Y, lr=1, threshold=1e-8, max_iteration=float('inf'), penalty="None", lmda=1, diminishing=False):
        self.X = X
        self.Y = Y
        self.n = X.shape[1]
        self.m = X.shape[0]
        self.W = np.zeros(self.n)
        self.B = 0
        self.lr = lr
        self.iteration_time = 0
        self.threshold = threshold
        self.max_iteration = max_iteration
        self.diminishing = diminishing
        if penalty == "None":
            self.penalty = none_penalty(self, lmda)
        elif penalty == "l1":
            self.penalty = l1_penalty(self, lmda)
        elif penalty == "l2":
            self.penalty = l2_penalty(self, lmda)
        self.fit()

    def fit(self):
        prev_loss = -self.log_mle()
        while True:
            if self.iteration_time >= self.max_iteration:
                break
            grad_W, grad_B = self.gradient()
            if self.diminishing:
                self.lr = 2 / (2 + self.iteration_time)
            self.W += self.lr * grad_W
            self.B += self.lr * grad_B
            self.iteration_time += 1
            loss = -self.log_mle()
            if np.abs(prev_loss - loss) < self.threshold:
                break
            prev_loss = loss

    def gradient(self):
        grad_W = self.penalty()
        grad_B = 0
        for i in range(self.m):
            x = self.X[i]
            y = self.Y[i]
            prob_x = self.prob(x)
            # update W
            for j in range(self.n):
                xj = x[j]
                grad_W[j] += xj * ((y+1)/2 - prob_x)
            # update B
            grad_B += ((y+1)/2 - prob_x)
        return grad_W, grad_B

    def predict(self, X):
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x = X[i]
            prob_x = self.prob(x)
            result[i] = 1 if prob_x >= 0.5 else -1
        return result

    def projection(self, x):
        return np.inner(self.W, x) + self.B

    def prob(self, x):
        proj = self.projection(x)
        return np.exp(proj) / (1 + np.exp(proj))

    def log_mle(self):
        result = 0.0
        for i in range(self.X.shape[0]):
            xi = self.X[i]
            yi = self.Y[i]
            proj = self.projection(xi)
            result += ((yi+1) / 2)*proj - np.log(1+np.exp(proj))
        return result


if __name__ == "__main__":
    train_X, train_Y = read_file("train.data")
    valid_X, valid_Y = read_file("valid.data")
    test_X, test_Y = read_file("test.data")

    # lr = LogisticRegression(train_X, train_Y, penalty="None", lr=0.1, diminishing=False, threshold=0)

    lrs = [0.001, 0.01, 0.1, 1]
    lmdas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10]
    los1 = dict()
    accus1 = dict()
    for lrate in lrs:
        for lmda in lmdas:
            lr = LogisticRegression(train_X, train_Y, penalty="l1", lmda=lmda, lr=lrate, diminishing=True, threshold=0.0001)
            predict_valid_Y = lr.predict(valid_X)
            los1[str(lrate), str(lmda)] = lr
            accus1[str(lrate), str(lmda)] = [np.sum(predict_valid_Y == valid_Y) / valid_Y.size]
            print("lr1: " + str(accus1) + ", lmda1: " + str(lmda))

    lmdas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10]
    lrs2 = [0.001, 0.01, 0.1, 1]
    los2 = dict()
    accus2 = []
    for lrate in lrs:
        for lmda in lmdas:
            lr = LogisticRegression(train_X, train_Y, penalty="l2", lmda=lmda, lr=lrate, diminishing=False, threshold=0.0001)
            predict_valid_Y = lr.predict(valid_X)
            los2[str(lrate), str(lmda)] = lr
            accus2[str(lrate), str(lmda)] = [np.sum(predict_valid_Y == valid_Y) / valid_Y.size]
            print("lr2: " + str(accus1) + ", lmda2: " + str(lmda))