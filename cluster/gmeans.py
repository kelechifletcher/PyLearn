import numpy as np
from numpy import linalg
from scipy import stats
from cluster.kmeans import kmeans


_sig_lvl_dict = {15.0: 0, 10.0: 1, 5.0: 2, 2.5: 3, 1.0: 4}


class GMeans:
    def __init__(self, data, significance_level=5.0, metric='euclidean'):
        # defined attributes
        self.data = data
        self.significance_level = significance_level
        self.metric = metric

        # computed attributes
        self.k = None
        self.centroids = None
        self.partition_mask = None
        self.error = None

    def cluster(self):
        self.k, self.centroids, self.partition_mask, self.error = gmeans(self.data, self.significance_level)
        return self


def gmeans(data, significance_level=5.0):
    """
    Computes G-means clustering

    :param data:
    :param significance_level:
    :type data: ndarray
    :type significance_level: float
    :return:
    """

    # data shape
    n, d = data.shape

    # significance level index
    sig_lvl_index = _sig_lvl_dict[significance_level]

    # number of restarts
    restart = 0
    num_restart = 5

    # compute initial k-means
    prev_centroids = np.empty(shape=(0, 0))
    centroids, partition_mask, error = kmeans(data=data, k=1)

    while (centroids.shape[0] > prev_centroids.shape[0]) and (restart < num_restart):
        prev_centroids = centroids

        if _is_quality_partition(partition_mask):
            centroids = np.empty(shape=(0, d))
            centroid_split = _compute_centroid_split(data, prev_centroids, partition_mask)

            for i in range(prev_centroids.shape[0]):
                c = prev_centroids[i]
                (c1, c2), m, e = kmeans(data=data[partition_mask[i]], centroids=centroid_split[i])
                v = c1 - c2
                x = np.dot(data[partition_mask[i]], v) / linalg.norm(v) ** 2
                x = stats.zscore(x)
                a2, crit_val, sig_lvl = stats.anderson(x)

                if a2 > crit_val[sig_lvl_index]:
                    centroids = np.vstack(tup=(centroids, c1, c2))
                else:
                    centroids = np.vstack(tup=(centroids, c))

            centroids, partition_mask, error = kmeans(data=data, centroids=centroids)

        # restart if there's a bad partition
        else:
            restart += 1
            prev_centroids = np.empty(shape=(0, 0))
            centroids, partition_mask, error = kmeans(data=data, k=1)

    return centroids.shape[0], centroids, partition_mask, error


def _compute_centroid_split(data, centroids, partition_mask):
    """
    Computes centroid splits via principle component

    :param data:
    :param centroids:
    :param partition_mask:
    :return:
    """

    # data shape
    n, d = data.shape
    k = centroids.shape[0]
    split_centroids = np.empty(shape=(k, 2, d))

    for i in range(k):
        d = data[partition_mask[i]]
        l, s = _principle_component(d)
        m = s * np.sqrt((2 * l) / np.pi)
        c1, c2 = (centroids[i] + m, centroids[i] - m)
        split_centroids[i] = np.vstack(tup=(c1, c2))

    return split_centroids


def _principle_component(data):
    """
    Computes predominant principle component via power iteration

    :param data:
    :return:
    """

    # data shape
    n, d = data.shape

    # calculate covariance matrix
    a = np.cov(data.T)

    # initial eigenvector
    a_v = np.random.rand(d)

    # initial eigenvalue
    prev_l = np.inf
    l = _eigenvalue(a, a_v)

    # power iteration
    while np.abs(prev_l - l) > 0.01:
        a_v1 = a.dot(a_v)
        a_v = a_v1 / np.linalg.norm(a_v1)
        prev_l = l
        l = _eigenvalue(a, a_v)

    return l, a_v


def _eigenvalue(a, v):
    """
    Computes eigenvalue

    :param a:
    :param v:
    :return:
    """

    av = a.dot(v)
    return v.dot(av)


def _is_quality_partition(partition_mask):
    return True if np.min(np.count_nonzero(partition_mask, axis=1)) > 1 else False

