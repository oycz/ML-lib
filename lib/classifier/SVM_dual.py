import numpy as np
import shelve
from cvxopt import matrix, solvers
from collections import Counter
from lib.preprocessing.read_data import read_file


# soft margin SVM (dual from with gaussian kernel)
class SVM:

    def __init__(self, file_name, c=1.0, sigma2=1, precision=1e-05, no_kernel=False):
        self.file_name = file_name
        self.c, self.sigma2 = c, sigma2
        self.sol = None
        self.X, self.Y = SVM.read_file(file_name)
        self.precision = precision
        if not no_kernel:
            self.kernel = SVM.gauss_kernel_generator(sigma2)
        else:
            self.kernel = np.inner
        self.lmda = None
        self.sv, self.sv_label, self.sv_index, self.sv_lmda = None, None, None, None
        self.B = 0
        # self.sv_bs, self.b = None, 0
        # self.W, self.B = None, None
        self.fit()

    @staticmethod
    def kernel_matrix_key(file_name, sigma2):
        return file_name + "..." + str(sigma2)

    @staticmethod
    def b_key(file_name, c, sigma2, precision):
        return file_name + "..." + str(c) + "..." + str(sigma2) + "..." + str(precision)

    # cache of kernel constants matrix of same sigma and same filename
    @staticmethod
    def get_kernel_matrix(file_name, X, Y, sigma2):
        K = np.zeros((X.shape[0], X.shape[0]))
        kernel = SVM.gauss_kernel_generator(sigma2)
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i, j] = kernel(X[i], X[j])
        P = matrix(np.outer(Y, Y) * K)
        return P

    def fit(self):
        # P
        P = SVM.get_kernel_matrix(self.file_name, self.X, self.Y, self.sigma2)

        # q
        q = -np.ones(self.X.shape[0])
        q = matrix(q)

        # G
        G = np.zeros((self.X.shape[0] * 2, self.X.shape[0]))
        for i in range(self.X.shape[0]):
            G[i][i] = -1
        for i in range(self.X.shape[0], 2 * self.X.shape[0]):
            G[i][i - self.X.shape[0]] = 1
        G = matrix(G)

        # h
        h = []
        for i in range(self.X.shape[0]):
            # h = h + [self.c]
            h = h + [0.0]
        for i in range(self.X.shape[0]):
            # h = h + [0.0]
            h = h + [self.c]
        h = matrix(h)

        # A
        A = np.zeros((1, self.X.shape[0]))
        for i in range(self.X.shape[0]):
            A[0][i] = self.Y[i]
        A = matrix(A)

        b = matrix(0.0)
        self.sol = solvers.qp(P, q, G=G, h=h, A=A, b=b)
        self.lmda = np.ravel(np.array(self.sol['x']))

        # find support vector index
        sv_index = []
        for l in self.lmda:
            if l >= self.precision:
                sv_index += [True]
            else:
                sv_index += [False]
        self.sv_index = np.array(sv_index)
        self.sv, self.sv_label, self.sv_lmda = self.X[self.sv_index], self.Y[self.sv_index], self.lmda[self.sv_index]
        self.B = self.calc_B()

    def calc_B(self):
        b_projects = self.project(self.sv)
        b_projects_count = Counter(b_projects).most_common()[0][0]
        # return 1 - b_projects_count
        return 1 - np.mean(b_projects)

    def project(self, X):
        projects = np.zeros(X.shape[0])
        for a in range(X.shape[0]):
            pa = np.zeros(self.X.shape[0])
            for i in range(self.X.shape[0]):
                xi = self.X[i]
                yi = self.Y[i]
                lmdai = self.lmda[i]
                pa[i] = lmdai * yi * self.kernel(X[a], xi)
            projects[a] = sum(pa)
        return projects + self.B

    def predict(self, X):
        return np.sign(self.project(X))


    @staticmethod
    def gradient(X, Y, W, B, c):
        grad_W, grad_B = np.array(W), 0
        hinge_grad = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            xi = X[i]
            yi = Y[i]
            for j in range(X.shape[1]):
                if yi * xi[j] * W[j] < 1:
                    hinge_grad[j] = hinge_grad[j] - yi * xi[j]
            if yi * xi[j] < 1:
                grad_B = grad_B - yi
        hinge_grad = c * hinge_grad
        grad_W = grad_W + hinge_grad
        grad_B = c * grad_B
        return grad_W, grad_B

    @staticmethod
    def loss(X, Y, W, B, c):
        nm = (1/2) * np.linalg.norm(W)
        hinge = 0
        for i in range(X.shape[0]):
            xi = X[i]
            yi = Y[i]
            u = 1 - yi * (np.inner(W, xi)) + B
            hinge = hinge + max(0, u)
        hinge = c * hinge
        return nm + hinge

    @staticmethod
    def linear_kernel_generator():
        def linear_kernel(x1, x2):
            return np.inner(x1, x2)
        return linear_kernel

    @staticmethod
    def gauss_kernel_generator(sigma2):
        def gauss_kernel(x1, x2):
            return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2. * sigma2))
        return gauss_kernel


def distance(hyperplane, point):
    return np.inner(hyperplane.flatten(), point) / np.linalg.norm(hyperplane[1:-1])


if __name__ == "__main__":
    # read data
    train_X, train_Y = read_file("train.data")
    validation_X, validation_Y = read_file("validation.data")
    test_X, test_Y = read_file("test.data")

    train_len = train_X.shape[0]
    validation_len = validation_X.shape[0]
    test_len = test_X.shape[0]
    #
    c_arr = [1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8]
    sigma2_arr = [0.1, 1, 10, 100, 1000]
    #
    s = dict()
    for c in c_arr:
        for sigma2 in sigma2_arr:
            # s[c, sigma2] = SVM("spam_train.data", c=c, sigma2=sigma2)
            print("c: ", c, " sigma2: ", sigma2)
            s[c, sigma2] = SVM("train.data", c=c, sigma2=sigma2, no_kernel=False)

    ## training
    accu_training = dict()
    pred_training = dict()
    for c in c_arr:
        for sigma2 in sigma2_arr:
            pred_training[c, sigma2] = s[c, sigma2].predict(train_X)
            accu_training[c, sigma2] = np.sum(pred_training[c, sigma2] == train_Y) / train_len

    ## validation
    accu_validation = dict()
    pred_validation = dict()
    for c in c_arr:
        for sigma2 in sigma2_arr:
            pred_validation[c, sigma2] = s[c, sigma2].predict(validation_X)
            accu_validation[c, sigma2] = np.sum(pred_validation[c, sigma2] == validation_Y) / validation_len

    ## test
    accu_test = dict()
    pred_test = dict()
    for c in c_arr:
        for sigma2 in sigma2_arr:
            pred_test[c, sigma2] = s[c, sigma2].predict(test_X)
            accu_test[c, sigma2] = np.sum(pred_test[c, sigma2] == test_Y) / test_len
