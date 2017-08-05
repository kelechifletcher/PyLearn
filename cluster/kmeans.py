import numpy as np
from scipy.spatial import distance


class KMeans:
    def __init__(self, data, k=None, centroids=None, metric='euclidean'):
        # defined attributes
        self.data = data
        self.k = k
        self.metric = metric

        # computed attributes
        self.centroids = centroids
        self.partition_mask = None
        self.error = None

    def cluster(self):
        self.centroids, self.partition_mask, self.error = kmeans(self.data, self.k, self.centroids, self.metric)
        return self


def kmeans(data, k=None, centroids=None, metric='euclidean'):
    """
    Computes k-means++ clustering

    :param data: data points
    :param k: number of clusters
    :param centroids: initial centroids
    :param metric: distance metric; euclidean or cosine
    :return: centroids: centroids of each computed partition
             partition_mask: boolean mask for each computed partition
             error: within-cluster error
    """

    if metric == 'euclidean' or metric == 'cosine':
        # initial centroids
        if centroids is None:
            if k:
                centroids = data[_initial_centroids(data, k)]
            else:
                raise AttributeError('must provide \'k\' or \'centroids\'')

        # initial partitions
        partition_mask, partition_error = _compute_voronoi_partition(data, centroids, metric)

        # initial error
        prev_error = np.inf
        error = partition_error

        while error < prev_error:
            # compute centroids
            temp_centroids = _compute_centroids(data, partition_mask)

            # compute clusters
            temp_mask, temp_error = _compute_voronoi_partition(data, temp_centroids, metric)

            # compute error
            prev_error = error
            if temp_error < error:
                centroids = temp_centroids
                partition_mask, partition_error = temp_mask, temp_error
                error = temp_error

    else:
        raise AttributeError('distance metric \'%s\' not supported' % metric)

    return centroids, partition_mask, error


def _initial_centroids(data, k, metric='euclidean'):
    """
    Randomly sample k points from data as initial centroids

    :param data:
    :param k:
    :param metric:
    :return:
    """

    if metric == 'euclidean' or metric == 'cosine':
        # data shape
        n, d = data.shape

        # initialize centroid mask
        centroid_mask = np.zeros(shape=n, dtype=bool)
        centroid_mask[np.random.randint(n)] = 1
        curr_k = 1

        # randomly select k points
        while curr_k < k:
            # compute distances between data points and centroids
            cdist_matrix = distance.cdist(data[~centroid_mask], data[centroid_mask], metric)

            # compute distance-based probability distribution
            cdist_prob = np.min(cdist_matrix, axis=1)
            cdist_prob /= np.sum(cdist_prob)

            # randomly select point from probability distribution
            centroid_mask[np.random.choice(np.where(~centroid_mask)[0], p=cdist_prob)] = 1
            curr_k += 1
    else:
        raise AttributeError('distance metric \'%s\' not supported' % metric)

    return np.nonzero(centroid_mask)


def _compute_centroids(data, partition_mask):
    """
    Computes centroids of voronoi partitions give a partition mask.

    :param data:
    :param partition_mask:
    :return:
    """

    # data shape
    n, d = data.shape
    k = partition_mask.shape[0]
    centroids = np.empty(shape=(k, d))

    for i in range(k):
        centroids[i] = np.mean(data[partition_mask[i]], axis=0)

    return centroids


def _compute_voronoi_partition(data, centroids, metric='euclidean'):
    """
    Computes voronoi partition of data from k centroids

    :param data:
    :param centroids:
    :param metric:
    :return:
    """

    if metric == 'euclidean' or metric == 'cosine':
        # data shape
        n, d = data.shape
        k = centroids.shape[0]

        # calculate distance between each data point and centroid
        cdist_matrix = distance.cdist(data, centroids, metric)

        # compute the closest centroid from each data point
        cdist_argmin = np.argmin(cdist_matrix, axis=1)

        # build partition mask
        partition_mask = np.zeros(shape=(k, n), dtype=bool)
        partition_mask[cdist_argmin, np.arange(n)] = 1

        # compute error
        if metric == 'euclidean':
            partition_error = np.sum(cdist_matrix[np.arange(n), cdist_argmin] ** 2)

        else:
            partition_error = np.sum(cdist_matrix[np.arange(n), cdist_argmin])

    else:
        raise AttributeError('distance metric \'%s\' not supported' % metric)

    return partition_mask, partition_error
