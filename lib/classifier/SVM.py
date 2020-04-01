import numpy as np
from cvxopt import matrix, solvers


# soft margin SVM (Primal)
class SVM:

    def __init__(self, X, Y, c=1.0):
        self.X = X
        self.Y = Y
        self.W, self.B = None, None
        self.count = 0
        self.c = c
        self.sol = None
        self.fit()

    def fit(self):
        m = self.X.shape[0]
        n = self.X.shape[1]

        # P
        P = np.zeros((n+m+1, n+m+1))
        for i in range(n):
            P[i][i] = 1
        P = matrix(P)

        # q
        q = np.zeros(n+m+1)
        for i in range(n+1, n+m+1):
            q[i] = self.c
        q = matrix(q)

        # G
        G = np.zeros((2*m, m+n+1))
        for i in range(m):
            xi = self.X[i]
            yi = self.Y[i]
            # top half
                # w
            for j in range(n):
                G[i][j] = -yi*xi[j]
                # b
            G[i][n] = -yi
                # xi
            G[i][n+1+i] = -1
            # bottom half
            G[m+i][n+i+1] = -1
        G = matrix(G)
        # h
        h = np.zeros(2*m)
        for i in range(m):
            h[i] = -1
        h = matrix(h)

        self.sol = solvers.qp(P, q, G=G, h=h)
        x = np.array(self.sol['x']).ravel()
        W_num = x.size - self.X.shape[0] - 1
        self.W = x[0: W_num]
        self.B = x[W_num]

    def predict(self, X):
        return np.sign(SVM.project(X, self.W, self.B))

    @staticmethod
    def project(X, W, B):
        return np.dot(X, W) + B

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
        return X, Y


def test_accuracy(X_to_test, Y_to_test, SVMs):
    test_accus = []
    for i in range(len(cs)):
        s = SVMs[i]
        pred_testX = s.predict(X_to_test)
        accu = np.sum(pred_testX == Y_to_test) / Y_to_test.size
        test_accus += [accu]
    return test_accus


if __name__ == "__main__":
    # read data
    train_X, train_Y = SVM.read_file("spam_train.data")
    validation_X, validation_Y = SVM.read_file("spam_validation.data")
    test_X, test_Y = SVM.read_file("spam_test.data")

    cs = [1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    SVMs = []
    pred_accus = []
    for c in cs:
        s = SVM(train_X, train_Y, c=c)
        SVMs += [s]
        pred_Y = s.predict(test_X)
        accu = np.sum(pred_Y == test_Y) / test_Y.size
        pred_accus += [accu]
        # print(c, ": ", accu)

    print("training: ")
    trainings = test_accuracy(train_X, train_Y, SVMs)
    for i in range(len(cs)):
        print(str(cs[i]) + ": " + str(trainings[i]))

    print("validation: ")
    validations = test_accuracy(validation_X, validation_Y, SVMs)
    for i in range(len(cs)):
        print(str(cs[i]) + ": " + str(validations[i]))

    print("test:  ")
    tests = test_accuracy(test_X, test_Y, SVMs)
    for i in range(len(cs)):
        print(str(cs[i]) + ": " + str(tests[i]))

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
    data = open("train.data", "r+", encoding="UTF-8-sig")
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

