import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from lib.preprocessing.read_data import read_file


class GMM:
    def __init__(self, X, k, mu=None, max_iteration=5, threshold=5):
        self.X = X
        self.m, self.n = X.shape[0], X.shape[1]
        self.k = k
        if mu is None:
            self.mu = np.random.random((k, self.n))
        else:
            self.mu = mu
        self.cov = np.array([None] * k)
        for i in range(k):
            self.cov[i] = np.eye(self.n)
        self.lmda = np.array([1/k] * k)
        self.Q = None
        self.max_iteration = max_iteration
        self.iteration = 0
        self.threshold = threshold
        self.fit()

    def fit(self):
        prev_loglike = float('-inf')
        while True:
            # E-step
            # add random noise to avoid overflow
            self.Q = np.random.normal(1e-7, 1e-7, (self.X.shape[0], k))
            for i in range(self.m):
                xi = self.X[i]
                s = 0.0
                for y in range(self.k):
                    self.Q[i][y] = self.lmda[y] * self.pdf_y(xi, y)
                    s = np.add(s, self.Q[i][y], dtype=np.float64)
                    # s += self.Q[i][y]
                self.Q[i] = self.Q[i] / s

            # M-step
            # # update mu
            for y in range(self.k):
                u = 0.0
                for i in range(self.m):
                    u += self.Q[i][y] * self.X[i]
                self.mu[y] = u / np.sum(self.Q[:, y])
            # # update cov
            for y in range(self.k):
                u = np.zeros((self.n, self.n))
                for i in range(self.m):
                    t = np.asmatrix(self.X[i] - self.mu[y]).T
                    u += self.Q[i][y] * np.dot(t, t.T)
                # self.cov[y] = (u / np.sum(self.Q[:, y])) + np.identity(self.n)
                self.cov[y] = (u / np.sum(self.Q[:, y]))
                diagonalize(self.cov[y])
            # # upadte lmda
            for y in range(self.k):
                self.lmda[y] = np.sum(self.Q[:, y]) / self.m
            self.iteration += 1
            loglike = self.log_likelihood()
            if self.iteration >= self.max_iteration or np.abs(loglike-prev_loglike)<self.threshold:
                break
            prev_loglike = loglike
            print(loglike)

    def pdf_y(self, x, y):
        return multivariate_normal.pdf(x, mean=self.mu[y], cov=self.cov[y], allow_singular=True)

    def pred(self, x):
        ys_prob = np.zeros(self.k)
        for y in range(self.k):
            ys_prob[y] = self.pdf_y(x, y)
        return np.argmax(ys_prob)

    def pred_all(self, X):
        rst = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            rst[i] = self.pred(X[i])
        return rst

    def log_likelihood(self):
        s = 0.0
        for xi in self.X:
            t = 0.0
            for y in range(self.k):
                t += self.lmda[y] * self.pdf_y(xi, y)
            s += np.log(t)
        return s

    @staticmethod
    def gaussian_pdf(x, mu, cov):
        n = len(x)
        t = np.array(x - mu).reshape(1, n)
        exp = np.exp(-0.5 * np.dot(np.dot(t, np.linalg.inv(cov)), t.T))
        return (1 / np.sqrt(((2 * np.pi) ** n) * np.linalg.det(cov))) * exp


def diagonalize(A):
    eigv, P = np.linalg.eigh(A)
    diag = np.linalg.multi_dot((P.T, A, P))
    return diag


def kmeanspp_init(X, k):
    m, n = X.shape
    b = np.array([False] * X.shape[0])
    r = np.random.random_integers(0, X.shape[0]-1)
    centres_idx = [r]
    b[r] = True
    for y in range(k-1):
        ds = np.array([0.0] * m)
        for i in range(m):
            xi = X[i]
            if b[i]:
                continue
            ds[i] = np.min(np.linalg.norm((X[centres_idx] - xi), axis=1))
        p = ds / np.sum(ds)
        new_r = np.random.choice(m, p=p)
        centres_idx += [new_r]
        b[new_r] = True
    return X[centres_idx]


def loss_kmeans(X, mus, labels):
    s = 0.0
    for i in range(X.shape[0]):
        xi = X[i]
        labeli = labels[i]
        s += np.sum((xi - mus[labeli])**2)
    return s


def func_uniform_draw(left, right):
    def draw(m, n):
        return np.random.uniform(left, right, (m, n))
    return draw


if __name__ == "__main__":
    train_X, train_Y = read_file("train.data")
    func_draw = func_uniform_draw(-3, 3)
    ks = [12, 18, 24, 36, 42]

    # k-means
    k1, g1 = dict(), dict()
    for k in ks:
        print(k)
        k1[k], g1[k] = [], []
        for i in range(20):
            init_centres = func_draw(k, train_X.shape[1])
            k1[k] += [KMeans(n_clusters=k, init=init_centres).fit(train_X)]
            g1[k] += [GMM(train_X, k, mu=init_centres)]

    # k-means++, GMM++
    k2, g2 = dict(), dict()
    for k in ks:
        print(k)
        k2[k], g2[k] = [], []
        for i in range(20):
            init_centres = kmeanspp_init(train_X, k)
            k2[k] += [KMeans(n_clusters=k, init=init_centres).fit(train_X)]
            g2[k] += [GMM(train_X, k, mu=init_centres)]

    # evaluate k-means
    k1_loss, k2_loss = dict(), dict()
    for k in ks:
        # print(k)
        k1_loss[k], k2_loss[k] = [], []
        for m in k1[k]:
            c, l = m.cluster_centers_, m.labels_
            k1_loss[k] += [loss_kmeans(train_X, c, l)]
        for m in k2[k]:
            c, l = m.cluster_centers_, m.labels_
            k2_loss[k] += [loss_kmeans(train_X, c, l)]

    # evaluate GMM
    g1_loglike, g2_loglike = dict(), dict()
    for k in ks:
        # print(k)
        g1_loglike[k], g2_loglike[k] = [], []
        for g in g1[k]:
            g1_loglike[k] += [g.log_likelihood()]
        for g in g2[k]:
            g2_loglike[k] += [g.log_likelihood()]

    k1_means = dict()
    k1_vars = dict()
    k2_means = dict()
    k2_vars = dict()
    g1_means = dict()
    g1_vars = dict()
    g2_means = dict()
    g2_vars = dict()
    for k in ks:
        k1_means[k] = np.mean(k1_loss[k])
        k1_vars[k] = np.var(k1_loss[k])
        k2_means[k] = np.mean(k2_loss[k])
        k2_vars[k] = np.var(k2_loss[k])

        g1_means[k] = np.mean(g1_loglike[k])
        g1_vars[k] = np.var(g1_loglike[k])
        g2_means[k] = np.mean(g2_loglike[k])
        g2_vars[k] = np.var(g2_loglike[k])

    kmeans_36_l = k1[36][0].labels_
    g1_36_l = g1[36][0].pred_all(train_X)
