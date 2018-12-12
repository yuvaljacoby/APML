import pickle

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE


def plot_tsne(data, y):
    embedded = TSNE(n_components=2).fit_transform(data)
    # pickle.dump(embaded, open("tsne_mnist.pickle", "wb"))
    # embaded = pickle.load(open("tsne_mnist.pickle", "rb"))

    plt.scatter(embedded[:, 0], embedded[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.show()


if __name__ == "__main__":
    digits, y_digits = load_digits(return_X_y=True)
    plot_tsne(digits, y_digits)
