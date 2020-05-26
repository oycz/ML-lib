import numpy as np
from collections import Counter
from scipy import stats
from lib.preprocessing.read_data import read_file


class GaussianNaiveBayes:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.mus = dict()
        self.sigma2s = dict()
        self.ys = [y[0] for y in Counter(self.Y).most_common()]
        self.fit()

    def fit(self):
        for y in self.ys:
            self.calc_mu(y)
            self.calc_sigma2(y)

    def calc_mu(self, y):
        self.mus[y] = np.zeros(self.n)
        X = self.X[self.Y == y]
        for j in range(self.n):
            self.mus[y][j] = np.mean(X[:, j])

    def calc_sigma2(self, y):
        self.sigma2s[y] = np.zeros(self.n)
        X = self.X[self.Y == y]
        for j in range(self.n):
            muj = self.mus[y][j]
            self.sigma2s[y][j] = np.mean((X[:, j]-muj)**2)

    def predict(self, X):
        predicts = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            predicts[i] = self.predict_unit(X[i])
        return predicts

    def predict_unit(self, x):
        y_log_probs = []
        for y in self.ys:
            y_log_prob = 0.0
            for j in range(self.n):
                xj = x[j]
                # prob = stats.norm.pdf(xj, self.mus[y][j], self.sigma2s[y][j])
                prob = stats.norm.pdf(xj, self.mus[y][j], np.sqrt(self.sigma2s[y][j]))
                y_log_prob += np.log(prob)
            y_log_probs += [y_log_prob]
        return self.ys[np.argmax(y_log_probs)]


if __name__ == "__main__":
    train_X, train_Y = read_file("train.data")
    valid_X, valid_Y = read_file("valid.data")
    test_X, test_Y = read_file("test.data")

    gnb = GaussianNaiveBayes(train_X, train_Y)
    pred_valid_Y = gnb.predict(valid_X)
    prev_test_Y = gnb.predict(test_X)
    accu1 = np.sum(pred_valid_Y == valid_Y) / valid_Y.size
    accu2 = np.sum(prev_test_Y == test_Y) / test_Y.size
