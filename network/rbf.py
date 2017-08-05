import numpy as np
from numpy import linalg
from scipy.spatial import distance
from cluster.kmeans import KMeans
from cluster.gmeans import GMeans


class RBFNetwork:
    def __init__(self, data, labels, k=None, radius=None, metric='euclidean'):
        # defined attributes
        self.data = data
        self.labels = labels
        self.k = k
        self.radius = radius
        self.metric = metric

        # computed attributes
        if k:
            cluster = KMeans(data, k).cluster()
        else:
            cluster = GMeans(data).cluster()
            self.k = cluster.k

        self.centroids = cluster.centroids
        self.weights = None

        # compute heuristic for radius if none provided
        if not radius:
            n, d = self.data.shape
            centroid = np.mean(data, axis=0)
            self.radius = np.max(distance.cdist(centroid[np.newaxis, :], data)) / (k ** (1 / d))

    def fit(self):
        # declare variables
        n, d = self.data.shape
        x = self.data
        y = self.labels[np.newaxis, :]
        c = self.centroids
        r = self.radius

        # compute feature matrix
        z = np.hstack((np.ones(shape=(n, 1)), np.matrix(_gaussian_window(distance.cdist(x, c), r))))

        # compute psuedo-inverse
        self.weights = np.squeeze(np.array(linalg.inv(z.T * z) * z.T * y.T))

    def predict(self, x):
        if x.ndim < 2:
            x = x[np.newaxis, :]

        # declare variables
        n, d = x.shape
        c = self.centroids
        w = self.weights[np.newaxis, :]
        r = self.radius

        z = np.hstack((np.ones(shape=(n, 1)), np.matrix(_gaussian_window(distance.cdist(x, c), r))))
        z = np.squeeze(np.array(z * w.T))

        return z


def _gaussian_window(d, r):
    return np.e ** ((1 / 2) * (d / r) ** 2)
