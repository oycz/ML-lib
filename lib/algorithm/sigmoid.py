import numpy as np


def logistic(X):
    return 1 / (1 + np.exp(-X))

