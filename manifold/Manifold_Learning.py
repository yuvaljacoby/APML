import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances


def digits_example():
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target
    return data,labels, digits

    # plot examples:
    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()


def swiss_roll_example():
    '''
    Example code to show you how to load the swiss roll data and plot it.
    '''

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def faces_example(path):
    '''
    Example code to show you how to load the faces data.
    '''

    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels ** 0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.show()


def plot_with_images(X, images, title, image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels ** 0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig


def MDS(X, d):
    '''
    Given a NxD data matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxD data matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''

    delta = euclidean_distances(X, X)
    n = delta.shape[0]
    H = np.identity(n) - ((1 / n) * np.ones((n, n)))
    S = -0.5 * np.matmul(np.matmul(H, delta), H)
    Lambda, U = np.linalg.eig(S)
    Lambda = np.sqrt(Lambda[:d])
    U = U[:, :d]
    return Lambda * U


def knn(X, k):
    '''

    :param X: NxD data matrix.
    :param k: the number of neighbors
    :return: index of the neighbors
    '''
    dist = np.square(euclidean_distances(X))
    indices = np.argsort(dist)[:, 1:k + 1]
    return indices


def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''
    n = X.shape[0]
    # LLE step 1
    neighbors_mat = knn(X, k)

    # LLE step 2
    W = np.zeros((n, n))
    for j in range(n):
        j_vec = X[j, :]
        j_neighbors = neighbors_mat[j, :]
        z = j_vec - X[j_neighbors, :]
        G = np.dot(z, z.T)
        G_inv = np.linalg.pinv(G)
        G_inv_sum = np.sum(G_inv, axis=1)
        G_norm = np.divide(G_inv_sum, np.dot(np.ones(k), G_inv_sum))
        W[j, j_neighbors] = G_norm

    # LLE step 3
    M = np.identity(n) - W
    MM = np.matmul(M.T, M)
    eigenvalues, eigenvectors = np.linalg.eigh(MM)
    return eigenvectors[:, 1 : (d+1)]


def DiffusionMap(X, d, sigma, t):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    kernel matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the kernel matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :return: Nxd reduced data matrix.
    '''

    # TODO: YOUR CODE HERE

    pass


def plot_embedding(X, y, digits, title=None):
    from matplotlib import offsetbox

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):

        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)



if __name__ == '__main__':
    digits_x, digits_y, digits = digits_example()
    # digits_embedded = MDS(digits_x, 2)
    # mds_fig = plot_with_images(digits_embedded, digits, 'MDS digits 2d', image_num=45)
    # plt.show()

    k = 10
    digits_embedded = LLE(digits_x, 2, k=k)
    plot_embedding(digits_embedded, digits_y, digits, "LLE digitis k=", k)
    # plot_with_images(digits_embedded, digits, 'LLE digits 2d', image_num=45)
    plt.show()
    pass
