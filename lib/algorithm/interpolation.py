import numpy as np


# interpolation as default gradient method （generator function）
# TODO test
def gen_interpolation_gradient(func, eps=0.000001):
    def gradient(X):
        grad = np.zeros(X.shape)
        for Xi in range(X.shape[0]):
            X1 = X.copy()
            X1[Xi] -= eps
            X2 = X.copy()
            X2[Xi] += eps
            val1 = func(X1)
            val2 = func(X2)
            grad[Xi] = (val2 - val1) / (2 * eps)
        return grad.T
    return gradient
