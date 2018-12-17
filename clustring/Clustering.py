import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize


def gaussian_data(k=4, dim=5, plot=False):
    num_points = 100  # per cluster
    data = np.empty((num_points * k, dim))
    std_scale = 0.1
    mean_scale = 10
    for i in range(k):
        for j in range(dim):
            std = np.random.random(1) * std_scale
            mean = np.random.random(1) * mean_scale
            data[num_points * i: num_points * (i + 1), j] = np.random.normal(mean,
                                                                             std,
                                                                             num_points)
    np.random.shuffle(data)
    if dim == 2 and plot:
        plt.scatter(data[:, 0], data[:, 1])
        plt.show()
    return data


def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.array([np.cos(t) + 0.1 * np.random.randn(length),
                        np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.array([2 * np.cos(t) + 0.1 * np.random.randn(length),
                        2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.array([3 * np.cos(t) + 0.1 * np.random.randn(length),
                        3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.array([4 * np.cos(t) + 0.1 * np.random.randn(length),
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
    return apml
    # plt.plot(apml[:, 0], apml[:, 1], '.')
    # plt.show()


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

    return data
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


def cost_l2(X, center):
    """
    return the diameter of X
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: The cost
    """

    dist = euclid(X, center) ** 2
    # dist = get_upper_tringale(dist)
    return np.sum(dist)


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
    # prob_points = np.zeros((num_points, 1))
    centroids_idx[0] = np.random.choice(num_points, 1)
    min_dist = np.full((1, X.shape[0]), np.infty)
    for i in range(1, k):
        dist_to_last = metric(X[centroids_idx[i - 1], :][None, :], X)
        min_dist = np.minimum(dist_to_last, min_dist)
        prob_points = np.asarray((min_dist / np.sum(min_dist)).T)
        # make sure sum is 1
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
    k, d = centers.shape
    # Might happen that centers will be none in high dimensions space
    # centers = centers[~np.isnan(centers)].reshape(-1,d)
    dist = metric(X, centers)
    assignment = np.argmin(dist, axis=1)
    assert (assignment.shape[0] == X.shape[0])
    assert (np.max(assignment) <= centers.shape[0])
    return assignment


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init, cost_func=None):
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
        assignment = kmaens_assignment(X, centroids, metric)
        for i in range(k):
            centroids[i, :] = center(X[assignment == i, :])

    assignment, centers = (np.squeeze(np.asarray(assignment)), centroids)
    if not cost_func:
        return assignment, centers
    else:
        costs = np.average([cost_func(X[assignment == j], centers[j, :][None, :]) for j in range(centers.shape[0])])
        return assignment, centers, costs


def plot_clusters(X, assignment, title):
    clusters = np.unique(assignment)
    X = np.asarray(X)
    plt.scatter(X[:, 0], X[:, 1], c=assignment, cmap=plt.cm.get_cmap("rainbow", len(clusters)))
    plt.colorbar(ticks=range(len(clusters)))
    plt.title(title)
    plt.savefig("plots/" + title.replace(" ", "_").replace(".", ""))
    # plt.show()
    plt.clf()


def plot_four_clusters(X, assignments, titles):
    X = np.asarray(X)
    fig = plt.figure()
    for k in range(0,min(len(assignments), 4)):
        ax = fig.add_subplot(2,2,k+1)
        clusters = np.unique(assignments[k])
        ax.scatter(X[:, 0], X[:, 1], c=assignments[k], cmap=plt.cm.get_cmap("rainbow", len(clusters)))
        ax.set_title(titles[k])
    # plt.show()
    plt.savefig("plots/" + titles[0].replace(" ", "_"))
    plt.clf()


def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """

    return np.exp(- (X) / (2 * np.square(sigma)))


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

    return np.logical_or(neighbors, neighbors.T)


def spectral(X, k, similarity_param, similarity=gaussian_kernel, plot_similarity = False):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """

    N, d = X.shape
    S = euclid(X, X)
    W = similarity(S, similarity_param)

    D = np.diag(np.sum(W, axis=1))
    D_root = np.linalg.pinv(np.sqrt(D))  # D^-0.5
    L = np.eye(N, N) - np.dot(D_root, np.dot(W, D_root))
    eiganvalue, eiganvectors = np.linalg.eigh(L)

    valid_k = min(len(eiganvalue), k)
    if k != valid_k:
        print('log: Error, spectral clustering more clusters then eigan values')

    eiganvectors[:, np.where(eiganvalue < 0)] *= -1
    eiganvalue[np.where(eiganvalue < 0)] *= -1

    norm_eiganvectors = normalize(eiganvectors[:, np.argsort(eiganvalue)[:valid_k]])

    assignment, centers = kmeans(norm_eiganvectors, valid_k)

    if plot_similarity:
        title = "spectral clustering using %s and param: %.3f" % (similarity.__name__, similarity_param)
        shuffled_w = np.random.permutation(W)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(shuffled_w, cmap='hot')
        ax1.set_title('shuffled')
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(W[np.argsort(assignment)], cmap='hot')
        ax2.set_title('sorted')
        fig.suptitle(title)
        # plt.show()
        plt.savefig("plots/" + title.replace(" ", "_").replace(".", ""))
        plt.clf()

    return assignment, centers



def explore_k(X, algorithm, title, cost=cost_l2, search_range=range(1, 15)):
    costs = []
    for i in search_range:
        assignment, centers, cur_cost = algorithm(X, i, cost_func=cost)
        costs.append(cur_cost)
        print('log: param: %d cost: %.3f' % (i, costs[-1]))

    plt.scatter(list(search_range), costs)
    plt.xlabel('amount of clusters (k)')
    plt.ylabel('cost')
    plt.title(title)
    plt.savefig(('plots/' + title).replace(" ", "_"))
    # plt.show()
    plt.clf()


def get_upper_tringale(X):
    rows = X.shape[0]
    return (X.flatten())[:round(rows ** 2 / 2)]


def distances_histogram(X, title, percentile=5, plot=True):
    dist = euclid(X, X)
    # dist is symmetric --> take only the first half...
    dist = get_upper_tringale(dist)
    if plot:
        plt.hist(dist)
        plt.xlabel('distance')
        plt.ylabel('number of points')
        plt.title(title)
        plt.savefig("plots/" + title.replace(" ", "_"))
        # plt.show()
        plt.clf()
    return np.percentile(dist, [percentile])


def experiment_spectral(data, k, title, similarity_param_range=range(10, 100, 10), similarity=gaussian_kernel):
    assignments = []
    # similarity_param_range = similarity_param_range[:2]
    for p in similarity_param_range:
        assignment, centroids = spectral(data, k, p, similarity)
        # plot_clusters(data, assignment, title % (p))
        assignments.append(list(assignment))
        print('log: spectral clustring on param:', p)
        # centroids.append(centroids)

    plot_four_clusters(data, assignments, [title % p for p in similarity_param_range])


def experiment_mnn_m(data, data_name, k, m_range=range(5, 205, 50)):
    experiment_spectral(data, k, ("spectral " + data_name + " data, mnn: %d"), m_range, mnn)


if __name__ == '__main__':
    np.random.seed(42)

    ########## SECTION 2.5.1 - Choosing k, using the "elbow" method ###########
    # generate data for k = 20 clusters, we expect to see a drop in the cost around k=20.
    # The data is generated randomly from normal distribution, depends on the distributions we might see the
    # drop before...
    k = 20
    d = 2
    data = gaussian_data(k, d)
    explore_k(data, kmeans, "l2 cost kmeans on {} multivariate gaussians - distances histogram"
              .format(k,d), cost=cost_l2, search_range=range(5, 50))

    best_k = 15
    assignment, centroids = kmeans(data, best_k)
    plot_clusters(data, assignment, "kmeans multivariate gaussians data, k: %d" % (best_k))


    ######### Spectral clustring APML pic - Similarity Graph ###############
    # Find m
    k = 9  #prior knowledge
    data = apml_pic_example()
    experiment_mnn_m(data, 'APML pic', k)
    best_m = 55 # just trail and error

    # sigma we choose according to the precentile in the distance matrix
    sigma = distances_histogram(data, "APML pic - distances histogram", plot=True, percentile=5)
    assignment, centroids = spectral(data, k, sigma, plot_similarity=True)
    plot_clusters(data, assignment, "gaussian kernel APML pic data - sigma: %.3f" % (sigma))
    assignment, centroids = spectral(data, k, best_m, mnn, plot_similarity=True)
    plot_clusters(data, assignment, "mnn APML pic data - best_m: %d" % (best_m))

    ########### microarray ##############
    # data = microarray_exploration()
    # assignment, centroids = spectral(data, 10, 3)
    # plot_clusters(data, assignment, "kmeans clusters")
