import numpy as np
from cvxopt import matrix, solvers


# Hard margin SVM
class SVM:
    def __init__(self, X, Y, method="quadratic"):
        self.X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.Y = Y
        self.theta = np.zeros(self.X.shape[1])
        self.sol = None
        if method == "quadratic":
            self.quadratic_fit()
        else:
            raise Exception("method name error")

    def quadratic_fit(self):
        P = np.identity(self.X.shape[1])
        P[0][0] = 0
        q = np.zeros(self.X.shape[1])
        scaler_G = np.zeros((self.Y.size, self.Y.size))
        for i in range(self.Y.size):
            scaler_G[i][i] = -Y[i]
        G = np.dot(scaler_G, self.X)
        h = -np.ones(self.X.shape[0])
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        # put data in quadratic solver
        self.sol = solvers.qp(P, q, G=G, h=h)

    # todo
    def dual_fit(self):
        pass

def distance(hyperplane, point):
    return np.inner(hyperplane.flatten(), point) / np.linalg.norm(hyperplane[1:-1])


if __name__ == "__main__":
    # read data
    data = open("mystery.data", "r+", encoding="UTF-8-sig")
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

    s = SVM(X, Y)
    theta = np.array(s.sol['x'])
    print(theta)

    # test accuracy
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    rst = np.dot(X, theta)
    classify_rst = (np.sign(rst.T) == Y)
    # should be all true
    print(classify_rst)

    print("false rate (0 if perfectly classified): ", (classify_rst.size - np.count_nonzero(classify_rst))/classify_rst.size)

    # find support vector
    index = []
    deltas = []
    for i in range(rst.size):
        margin = rst[i][0]
        delta = np.abs(margin)-1
        # allow little precision inaccuracy
        deltas += [abs(delta)]
        if abs(delta) < 10e-10:
            index += [i]
    # print out index of support vector
    print(index)  # [7, 574, 969]

    support_vectors = X[index]

    # calculate margins (geometric margin between support vectors)
    margins = []
    for sv in support_vectors:
        margins += [distance(theta, sv)]

    print(margins)

