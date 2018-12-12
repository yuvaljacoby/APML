import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def gussians_data(k=4):
    num_points = 200
    data = np.empty((num_points * k, 2))
    mu_scale = 10
    std_scale = 2
    for i in range(k):
        data[num_points * i: num_points * (i + 1), 0] = np.random.normal(np.random.random(1) * mu_scale,
                                                                         np.random.random(1) * std_scale,
                                                                         num_points)
        data[num_points * i: num_points * (i + 1), 1] = np.random.normal(np.random.random(1) * mu_scale,
                                                                         np.random.random(1) * std_scale,
                                                                         num_points)

    np.random.shuffle(data)
    plt.scatter(data[:,0], data[:,1])
    plt.show()
    return np.matrix(data)


def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)

    # plt.plot(circles[0, :], circles[1, :], '.k')
    # plt.show()
    return circles.T


def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    plt.plot(apml[:, 0], apml[:, 1], '.')
    plt.show()


def microarray_exploration(data_path='microarray_data.pickle',
                           genes_path='microarray_genes.pickle',
                           conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    return np.matrix(data)
    print(conds)
    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5, 5], [-5, 5], 'r')
    plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()


def cost_diameter(X):
    """
    return the diameter of X
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: The cost
    """
    return 1

def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """

    return euclidean_distances(X, Y)


def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """

    return np.average(X, axis=0)


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """

    centroids_idx = np.empty(k, dtype=int)
    num_points = X.shape[0]
    prob_points = np.zeros((num_points, 1))
    centroids_idx[0] = np.random.choice(num_points, 1)
    min_dist = np.full((1, X.shape[0]), np.infty)
    for i in range(1, k):
        dist_to_last = metric(X[centroids_idx[i - 1], :], X)
        min_dist = np.minimum(dist_to_last, min_dist)
        prob_points = np.asarray((min_dist / np.sum(min_dist)).T)
        # don't choose same center twice..
        # prob_points[centroids_idx[:i]] = 0
        # make sure sum is 1
        # prob_points = prob_points / np.sum(prob_points)
        assert (np.abs(np.sum(prob_points) - 1) < 0.005)
        centroids_idx[i] = np.random.choice(num_points, 1, False, prob_points.flatten())
        # centroids_idx = np.append(centroids_idx, )
    return X[centroids_idx, :]


def kmaens_assignment(X, centers, metric=euclid):
    '''
    For each row find the nearest cluster according to the centers
    :param X: MxD matrix, each row is a point in a D dimension space
    :param centers: kxD each row is a cluster centroid in a D dimension space
    :param metric: metric function to find distance between row and center
    :return: Mx1 assignment of each row to one of the clusters
    '''
    dist = metric(X, centers)
    assignment = np.argmin(dist, axis=1)
    assert (assignment.shape[0] == X.shape[0])
    assert (np.max(assignment) <= centers.shape[0])
    return assignment


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :return: a tuple of (clustering, centroids)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """
    centroids = init(X, k, metric)
    for _ in range(iterations):
        assignment = np.squeeze(kmaens_assignment(X, centroids, metric))
        for i in range(k):
            centroids[i, :] = center(X[np.nonzero(assignment == i)[0], :])

    return (np.squeeze(np.asarray(assignment)), centroids)


def plot_clusters(X, assignment, title):
    clusters = np.unique(assignment)
    X = np.asarray(X)
    plt.scatter(X[:, 0], X[:, 1], c=assignment, cmap=plt.cm.get_cmap("rainbow", len(clusters)))
    plt.colorbar(ticks=range(len(clusters)))
    plt.title(title)
    # plt.savefig("plots/" + title.replace(" ", "_"))
    plt.show()
    plt.clf()


def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """

    return np.exp(-(X ** 2) / (2 * sigma ** 2))


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """

    N = X.shape[0]
    neighbors = np.empty((N, N))
    for i in range(N):
        neighbors[i, np.argsort(X[i, :])[1:m + 1]] = 1

    mutual_neighbors = np.logical_or(neighbors, neighbors.T)
    return neighbors


def spectral(X, k, similarity_param, similarity=gaussian_kernel):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """

    N = X.shape[0]
    S = euclid(X, X)
    W = similarity(S, similarity_param)
    # w_col_sum =
    D = np.diag(np.sum(W, axis=1))
    D_root = np.sqrt(D)
    L = np.eye(N, N) - np.matmul(D_root, np.matmul(W, D_root))
    eiganvalue, eiganvectors = np.linalg.eigh(L)

    norm_eiganvectors = eiganvectors[:, np.argsort(eiganvalue)[:k]]
    # TODO: Correct normaliztion???
    cols_sum = np.sum(norm_eiganvectors, axis=0)
    norm_eiganvectors /= np.repeat(cols_sum[:, None], N, axis=1).T
    return kmeans(np.matrix(norm_eiganvectors), k)


if __name__ == '__main__':
    # Y = np.matrix([[-40, -1200], [5, 1], [1, 3], [2, 5], [100, 100]])
    # microarray_exploration()
    k = 4
    # circles = circles_example()
    data = gussians_data(k)
    assignment, centroids = spectral(data, 4, 3)
    plot_clusters(data, assignment, "kmeans gussians data")
    # print(euclidean_centroid(X))

    # plot_clusters(Y, kmeans(Y, 3)[0])
    # data = microarray_exploration()
    # assignment, centroids = kmeans(data, 15)
    # plot_clusters(data, assignment, "kmeans clusters")
    # assignment, centroids = spectral(gussian, k, 10)
    # print('assignment per cluster:\n', np.asarray(np.unique(assignment, return_counts=True)).T)
    # plot_clusters(gussian, assignment, "spectral clusters")
    # print(euclid(X, Y))
