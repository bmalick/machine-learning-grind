
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

def get_iris():
    iris = sklearn.datasets.load_iris()
    X = iris["data"]
    y = iris["target"]
    return X, y

def plot_ellipsoid(ax, mu, sigma, color):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))

    x0, y0 = mu
    a, b = np.sqrt(sigma[0, 0]), np.sqrt(sigma[1, 1])
    theta = np.linspace(0, 2*np.pi, 100)
    x = x0 + a * np.cos(theta)
    y = y0 + b * np.sin(theta)
    plt.plot(x, y, color=color, linestyle="--")

def decision_boundary(ax, xbounds, ybounds, colors, model, N, margin=0.1):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))

    dx = (xbounds[1] - xbounds[0])/N
    dy = (ybounds[1] - ybounds[0])/N
    xx, yy = np.meshgrid(
        np.arange(xbounds[0] - margin, xbounds[1] + margin, dx),
        np.arange(ybounds[0] - margin, ybounds[1] + margin, dy)
    )

    y_hat = model.predict(np.c_[xx.ravel(), yy.ravel()])
    y_hat = y_hat.reshape(xx.shape)
    ax.contourf(xx, yy, y_hat, alpha=0.35, levels=2, colors=colors)

def accuracy(model, X, y):
    return (model.predict(X)==y).mean()

def logloss(model, X, y):
    y_one_hot = np.zeros((len(y), 3))
    for i, yi in enumerate(y):
        y_one_hot[i][yi] = 1
        
    proba_hat = model.predict_proba(X)
    return - (y_one_hot * np.log(proba_hat)).sum(axis=-1).mean()
