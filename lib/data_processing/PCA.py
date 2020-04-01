import numpy as np

from lib.classifier.SVM import SVM


def read_file(filename, positive=2):
    data = open(filename, "r+", encoding="UTF-8-sig")
    X, Y = [], []
    for line in data:
        arr_line = line.split("\n")[0].split(",")
        X_unit = arr_line[0:-1]
        X_unit = [float(n) for n in X_unit]
        Y_unit = float(arr_line[-1])

        # correct Y_unit
        if Y_unit == positive:
            Y_unit = 1
        else:
            Y_unit = -1
        X += [X_unit]
        Y += [Y_unit]

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def PCA(mat, k, refer_mat=None):
    if refer_mat is None:
        refer_mat = mat
    normalized_mat = normalize(mat, refer_mat)
    normalize_refer_mat = normalize(refer_mat)
    cov = np.cov(normalize_refer_mat, rowvar=False)
    eigval, eig = np.linalg.eigh(cov)
    idx = np.argsort(eigval)[::-1]
    idx = idx[:k]
    eigval_k, eig_k = eigval[idx], eig[:, idx]
    return np.dot(normalized_mat, eig_k)


def feature_select_distribution(mat, k, refer_mat=None):
    if refer_mat is None:
        refer_mat = mat
    normalized_mat = normalize(mat, refer_mat)
    cov = np.cov(normalized_mat, rowvar=False)
    eigval, eig = np.linalg.eigh(cov)
    idx = np.argsort(eigval)[::-1]
    idx = idx[:k]
    eigval_k, eig_k = eigval[idx], eig[:, idx]
    squared_eig_k = eig_k**2
    pi = np.sum(squared_eig_k, axis=1) / k
    return pi


def select_feature(X, idx):
    vectors = X.T
    # selected features vectors
    new_vectors = vectors[idx]
    # select features sample
    return new_vectors.T


def normalize(X, refer_X=None):
    if refer_X is None:
        refer_X = X
    mean, std = get_mean_std(refer_X)

    normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            normalized_X[i][j] = (X[i][j] - mean[j]) / std[j]
    return normalized_X


def get_mean_std(X):
    return np.mean(X, axis=0), np.std(X, axis=0)


if __name__ == "__main__":
    train_X, train_Y = read_file("train.data")
    valid_X, valid_Y = read_file("valid.data")
    test_X, test_Y = read_file("test.data")

    # error on the validation set of soft margin SVM
    ks1 = [1, 2, 3, 4, 5, 6]
    cs1 = [1.0, 10.0, 100.0, 1000.0]
    SVMs1 = dict()
    accus1 = dict()
    for k in ks1:
        for c in cs1:
            PCA_train_X = PCA(train_X, k)
            PCA_valid_X = PCA(valid_X, k, train_X)
            s = SVM(PCA_train_X, train_Y, c=c)
            pred_Y = s.predict(PCA_valid_X)
            accu = np.sum(valid_Y == pred_Y) / valid_Y.size
            SVMs1[k, c] = s
            accus1[k, c] = accu

    # on test set
    accus2 = dict()
    for k in ks1:
        for c in cs1:
            PCA_test_X = PCA(test_X, k, train_X)
            s = SVMs1[k, c]
            pred_Y = s.predict(PCA_test_X)
            accu = np.sum(test_Y == pred_Y) / test_Y.size
            accus2[k, c] = accu

    # origin
    origin_SVMs = dict()
    origin_accus = dict()
    for c in cs1:
        svm = SVM(train_X, train_Y, c=c)
        pred_Y = svm.predict(test_X)
        accu = np.sum(pred_Y == test_Y) / test_Y.size
        origin_SVMs[c] = svm
        origin_accus[c] = accu
