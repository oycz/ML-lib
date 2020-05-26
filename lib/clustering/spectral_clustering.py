import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from lib.preprocessing.read_data import read_file
import cv2


def laplacian_matrix(A):
    D = np.diag(np.sum(A, axis=0))
    return D - A


def similarity_matrix(mat, sigma):
    A = np.zeros((mat.shape[0], mat.shape[0]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            xi = mat[i]
            xj = mat[j]
            A[i][j] = np.exp(-(1/(2*(sigma**2)))*(np.linalg.norm(xi-xj)**2))
    return A


def spectral_clustering(mat, k, sigma):
    sim = similarity_matrix(mat, sigma)
    lap = laplacian_matrix(sim)
    eigval, eig = np.linalg.eigh(lap)
    idx = np.argsort(eigval)
    idx = idx[:k]
    B = eig[:, idx]

    kmeans_model = KMeans(n_clusters=k, random_state=np.random.randint(100)).fit(B)
    labels = kmeans_model.labels_

    return labels


def calc_loss(points, labels):
    idx_0, idx_1 = labels == 0, labels == 1
    points_0, points_1 = points[idx_0], points[idx_1]
    mu0, mu1 = np.mean(points_0, axis=0), np.mean(points_1, axis=0)
    t0, t1 = points_0 - mu0, points_1 - mu1
    norm20, norm21 = np.zeros(points_0.shape[0]), np.zeros(points_1.shape[0])
    for i in range(t0.shape[0]):
        norm20[i] = np.linalg.norm(t0[i])**2
    for i in range(t1.shape[0]):
        norm21[i] = np.linalg.norm(t1[i])**2
    return np.sum(norm20) + np.sum(norm21)


def show(mat, labels):
    ## plot k-means
    idx_0 = labels == 0
    idx_1 = labels == 1
    ### label 0
    mat_0 = mat[idx_0]
    plt.scatter(mat_0[:, 0], mat_0[:, 1], c="red")
    ### label 1
    mat_1 = mat[idx_1]
    plt.scatter(mat_1[:, 0], mat_1[:, 1], c="blue")
    plt.title("spectral clustering")
    plt.show()


def output_image(arr, m, n, filename):
    arr = [255 if i == 1 else 0 for i in arr]
    a = np.array(arr)
    a = a.reshape((m, n))
    cv2.imwrite(filename, a)


if __name__ == "__main__":
    #####################
    # read files
    colors = ["red", "blue"]
    k = 2
    circs_X = read_file("data.data")

    # Q1 circs
    #####################
    # regular k-means
    kmeans_model1 = KMeans(n_clusters=k, random_state=np.random.randint(100)).fit(circs_X)
    kmeans_labels1 = kmeans_model1.labels_
    # show(circs_X, kmeans_labels1)
    print(calc_loss(circs_X, kmeans_labels1))

    # spectral clustering
    # sigma1 = 0.01
    # sigma1 = 0.1
    # sigma1 = 1
    # sigma1 = 5
    sigma1 = 10
    spectral_labels1 = spectral_clustering(circs_X, k, sigma1)
    show(circs_X, spectral_labels1)
    print(calc_loss(circs_X, spectral_labels1))

    # # Q2 image
    ####################
    bw = np.array(cv2.imread("demo.jpg", cv2.IMREAD_GRAYSCALE))
    flattened_bw = bw.flatten().reshape(-1, 1).astype(int)

    # regular k-means
    kmeans_model2 = KMeans(n_clusters=k, random_state=np.random.randint(100)).fit(flattened_bw)
    kmeans_labels2 = kmeans_model2.labels_
    output_image(kmeans_labels2, bw.shape[0], bw.shape[1], "kmeans_demo.jpg")

    # spectral clustering
    sigma2s = [0.0001, 0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
    for sigma2 in sigma2s:
        spectral_labels2 = spectral_clustering(flattened_bw, k, sigma2)
        output_image(spectral_labels2, bw.shape[0], bw.shape[1], "spectral_demo_" + str(sigma2).replace(".", "_") + ".jpg")
