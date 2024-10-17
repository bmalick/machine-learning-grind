import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import pca


def dim_red(X, y, n_components, kernel, ax, title):
    model = pca.kPCA(n_components=n_components, kernel=kernel)
    model.fit(X)

    print("2 first eigen values = ", model.eigenvalues.tolist()[:2])
    w1 = model.eigenvectors[:, 0]
    w2 = model.eigenvectors[:, 1]
    print("1st eigen value: %.2f%%" % (model.eigenvalues[0] / model.eigenvalues.sum()*100))
    print("2nd eigen value: %.2f%%" % (model.eigenvalues[1] / model.eigenvalues.sum()*100))

    X_transformed = model.transform(X)
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.4, c=y)
    ax.set_title(title)

if __name__ == "__main__":
    N = 100
    X, y = sklearn.datasets.make_moons(N, random_state=42)

    X = X[:N]
    y = y[:N]
    y = np.array(y, dtype=int)

    plt.figure(figsize=(19.2, 10.8))

    ax = plt.subplot(2,2,1)
    sep = "#"*10
    print("%s Linear kernel PCA %s" % (sep, sep))
    dim_red(X=X, y=y, n_components=2, kernel=pca.linear_kernel, ax=ax, title="Kernel PCA with linear kernel on moons")
    print(sep); print()

    ax = plt.subplot(2,2,2)
    print("%s RBF kernel PCA %s" % (sep, sep))
    sigma = pca.get_sigma(15)
    dim_red(X=X, y=y, n_components=2, kernel=pca.radial_based_function(sigma), ax=ax, title="Kernel PCA with RBF kernel on moons")
    print(sep); print()

    ax = plt.subplot(2,2,3)
    print("%s Polynomial kernel PCA %s" % (sep, sep))
    gamma = 0.2
    r = -1
    d = 2
    dim_red(X=X, y=y, n_components=2, kernel=pca.polynomial_kernel(gamma,r,d), ax=ax, title="Kernel PCA with polynomial kernel on moons")
    print(sep); print()

    ax = plt.subplot(2,2,4)
    print("%s Sigmoid kernel PCA %s" % (sep, sep))
    gamma = 0.1
    r = 1
    dim_red(X=X, y=y, n_components=2, kernel=pca.sigmoid_kernel(gamma,r), ax=ax, title="Kernel PCA with sigmoid kernel on moons")
    print(sep); print()

    plt.show()
