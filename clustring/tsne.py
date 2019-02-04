import pickle

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_embedded(embedded, y, title):
    plt.scatter(embedded[:, 0], embedded[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.title(title)
    # plt.show()
    plt.savefig("plots/" + title.replace(" ", "_"))

def plot_tsne(data, y):
    embedded = TSNE(n_components=2).fit_transform(data)
    plot_embedded(embedded, y, 't-SNE embedding - mnist')


def plot_pca(data, y):
    p = PCA(n_components=2)
    embedded = p.fit_transform(data)
    plot_embedded(embedded, y, 'PCA embedding - mnist')


if __name__ == "__main__":
    digits, y_digits = load_digits(return_X_y=True)
    plot_tsne(digits, y_digits)
    plot_pca(digits, y_digits)
