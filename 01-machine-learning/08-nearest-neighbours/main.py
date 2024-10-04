import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import utils
from nearset_neighbours import NearestNeighbours


def k_nearest(k, ax1, ax2, colors = np.array(['#ff7f00', '#377eb8'])):
    X, y = utils.create_dataset()
    xbounds = (X[:,0].min(), X[:,0].max())
    ybounds = (X[:,1].min(), X[:,1].max())
    model = NearestNeighbours(k=k)
    model.fit(X, y)

    ax1.scatter(X[:, 0], X[:, 1], c=colors[y.astype(int)], s=5)
    title = "k-Nearest Neighbours - k = %d" % model.k
    print(title + " - custom data")
    ax1.set_title(title)
    utils.decision_boundary(ax1, xbounds, ybounds, colors, model, 100)

    X, y = sklearn.datasets.make_moons(100)
    xbounds = (X[:,0].min(), X[:,0].max())
    ybounds = (X[:,1].min(), X[:,1].max())
    model = NearestNeighbours(k=k)
    model.fit(X, y)
    ax2.scatter(X[:, 0], X[:, 1], c=colors[y.astype(int)], s=5)
    title = "k-Nearest Neighbours - k = %d" % model.k
    print(title + " - moon data")
    ax2.set_title(title)
    utils.decision_boundary(ax2, xbounds, ybounds, colors, model, 100)

def kernel_method(sigma, ax1, ax2, colors = np.array(['#ff7f00', '#377eb8'])):
    X, y = utils.create_dataset()
    xbounds = (X[:,0].min(), X[:,0].max())
    ybounds = (X[:,1].min(), X[:,1].max())
    model_with_kernel = NearestNeighbours(gaussian_kernel=True, sigma=sigma)
    model_with_kernel.fit(X, y)
    ax1.scatter(X[:, 0], X[:, 1], c=colors[y.astype(int)], s=5)
    ax1.scatter(X[:, 0], X[:, 1], c=colors[y.astype(int)], s=5)
    title = r"Nearest Neighbours - $\sigma = %s$" % model_with_kernel.sigma
    print(title + " - custom data")
    ax1.set_title(title)
    utils.decision_boundary(ax1, xbounds, ybounds, colors, model_with_kernel, 100)

    X, y = sklearn.datasets.make_moons(100)
    xbounds = (X[:,0].min(), X[:,0].max())
    ybounds = (X[:,1].min(), X[:,1].max())
    model_with_kernel = NearestNeighbours(gaussian_kernel=True, sigma=sigma)
    model_with_kernel.fit(X, y)
    ax2.scatter(X[:, 0], X[:, 1], c=colors[y.astype(int)], s=5)
    ax2.scatter(X[:, 0], X[:, 1], c=colors[y.astype(int)], s=5)
    title = r"Nearest Neighbours - $\sigma = %s$" % model_with_kernel.sigma
    print(title + " - moon data")
    ax2.set_title(title)
    utils.decision_boundary(ax2, xbounds, ybounds, colors, model_with_kernel, 100)




def main():
    figure, axes = plt.subplots(4,4, figsize=(20,15))
    axes = axes.ravel()
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])

    k_nearest(k=1, ax1=axes[0], ax2=axes[1])
    k_nearest(k=10, ax1=axes[2], ax2=axes[3])
    k_nearest(k=30, ax1=axes[4], ax2=axes[5])
    k_nearest(k=40, ax1=axes[6], ax2=axes[7])
    kernel_method(sigma=1, ax1=axes[8], ax2=axes[9])
    kernel_method(sigma=0.1, ax1=axes[10], ax2=axes[11])
    kernel_method(sigma=0.01, ax1=axes[12], ax2=axes[13])
    kernel_method(sigma=0.001, ax1=axes[14], ax2=axes[15])
    plt.savefig("nearest-neighours.jpg")    
    plt.show()

if __name__ == "__main__": main()
