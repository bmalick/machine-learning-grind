###########################
# Support Vector Machines #
###########################

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

from sklearn.model_selection import cross_val_score

def apply_cross_validation(model, inputs, labels):
    scores = cross_val_score(model, inputs, labels, cv=5)
    for i,s in enumerate(scores):
        print("fold %d: %f%%" % (i,1-s))
    print("risk: %f" % (1-scores.mean()))

def get_data(name: str, n: int = None):
    if name.startswith("iris"):
        iris = sklearn.datasets.load_iris()
        X = iris["data"][:, (2,3)]
        y = iris["target"]
        if name=="iris1":
            X = X[y!=2, :]
            y = y[y!=2]
        elif name=="iris2":
            # versicolor vs virginica
            X = X[y!=0, :]
            y = y[y!=0] - 1
    elif name=="moons":
        X, y = sklearn.datasets.make_moons(n_samples=n, noise=0.1)
    xbounds = [X[:, 0].min(), X[:, 0].max()]
    ybounds = [X[:, 1].min(), X[:, 1].max()]
    return X, y, xbounds, ybounds

def decision_boundary(ax, model, X, y, xbounds, ybounds, margin=0.1, n=100, colors=None):
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=50, linewidths=1, edgecolors="black", c="white", alpha=0.4)
    ax.scatter(X[:,0], X[:,1], c=np.array(["#1f77b4", "#ff7f0e"])[y.astype(int)], s=10)
    dx = (xbounds[1] - xbounds[0]) / n
    dy = (ybounds[1] - ybounds[0]) / n
    xx, yy = np.meshgrid(
        np.arange(xbounds[0]-margin, xbounds[1]+margin, dx),
        np.arange(ybounds[0]-margin, ybounds[1]+margin, dy))
    z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    ax.contour(xx, yy, z, levels=[-1, 0, 1],
               colors="black", alpha=0.4,
               linestyles=["solid", "dashed", "solid"],
               linewidths=[1,2,1])

def get_sigma(gamma: float) -> float: return 1/np.sqrt(gamma*2)
def get_gamma(sigma: float) -> float: return 1/sigma**2
def find_sigma(X: np.ndarray) -> float: return sklearn.metrics.euclidean_distances(X).mean()
