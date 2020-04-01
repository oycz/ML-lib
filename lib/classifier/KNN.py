import numpy as np
from collections import Counter


class KNN:
    def __init__(self, X, Y, k=5, normalization=False):
        self.mean, self.std = KNN.get_mean_std(X)
        if normalization:
            self.X = KNN.normalize(X, self.mean, self.std)
        else:
            self.X = X
        self.Y = Y
        self.k = k
        self.normalization = normalization

    def predict(self, X):
        if self.normalization:
            X = KNN.normalize(X, self.mean, self.std)
        predicts = np.zeros(X.shape[0])
        dists = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            xi = X[i]
            distances = []
            for j in range(self.X.shape[0]):
                xj = self.X[j]
                yj = self.Y[j]
                dist = KNN.distance(xi, xj)
                distances = distances + [(dist, yj)]
            distances.sort()
            distances = distances[0: self.k]
            labels = [x[1] for x in distances]
            predicts[i] = Counter(labels).most_common(1)[0][0]
            dists[i] = distances[0][0]
        self.dists = dists
        return predicts

    @staticmethod
    def distance(v1, v2):
        return np.inner(v1 - v2, v1-v2)
        # return np.linalg.norm(v1 - v2)

    @staticmethod
    def read_file(filename):
        data = open(filename, "r+", encoding="UTF-8-sig")
        X, Y = [], []
        for line in data:
            arr_line = line.split("\n")[0].split(",")
            X_unit = arr_line[0:-1]
            X_unit = [float(n) for n in X_unit]
            Y_unit = float(arr_line[-1])
            # correct Y_unit
            if Y_unit == 0:
                Y_unit = -1
            X += [X_unit]
            Y += [Y_unit]
        X = np.array(X)
        Y = np.array(Y)
        return np.array(X), np.array(Y)

    @staticmethod
    def normalize(X):
        normalized_X = np.zeros((X.shape[0], X.shape[1]))
        mean = np.zeros(X.shape[1])
        std = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            mean[j] = np.mean(X[:, j])
        for j in range(X.shape[1]):
            std[j] = np.std(X[:, j])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                normalized_X[i][j] = (X[i][j] - mean[j]) / std[j]
        return normalized_X

    @staticmethod
    def get_mean_std(X):
        return np.mean(X, axis=0), np.std(X, axis=0)

    @staticmethod
    def normalize(X, mean, std):
        normalized_X = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                normalized_X[i][j] = (X[i][j] - mean[j]) / std[j]
        return normalized_X


if __name__ == "__main__":
    # read data
    train_X, train_Y = KNN.read_file("train.data")
    validation_X, validation_Y = KNN.read_file("validation.data")
    test_X, test_Y = KNN.read_file("test.data")

    k1 = KNN(train_X, train_Y, k=1, normalization=True)
    k5 = KNN(train_X, train_Y, k=5, normalization=True)
    k11 = KNN(train_X, train_Y, k=11, normalization=True)
    k15 = KNN(train_X, train_Y, k=15, normalization=True)
    k21 = KNN(train_X, train_Y, k=21, normalization=True)
    pred_test_1 = k1.predict(test_X)
    pred_test_5 = k5.predict(test_X)
    pred_test_11 = k11.predict(test_X)
    pred_test_15 = k15.predict(test_X)
    pred_test_21 = k21.predict(test_X)

    test_len = test_X.shape[0]

    accu_1 = np.sum(pred_test_1 == -1) / test_len
    accu_5 = np.sum(pred_test_5 == -1) / test_len
    accu_11 = np.sum(pred_test_11 == -1) / test_len
    accu_15 = np.sum(pred_test_15 == -1) / test_len
    accu_21 = np.sum(pred_test_21 == -1) / test_len

    print("accu1, ", accu_1)
    print("accu_5, ", accu_5)
    print("accu_11, ", accu_11)
    print("accu_15, ", accu_15)
    print("accu_21, ", accu_21)
