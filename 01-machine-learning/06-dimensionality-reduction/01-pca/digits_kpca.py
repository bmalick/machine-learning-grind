import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import pca

def dim_red(X, y, n_components, kernel, ax, title):
    model = pca.kPCA(n_components=n_components, kernel=kernel)
    model.fit(X)

    print("2 first eigen values = ", model.eigenvalues.tolist()[:2])
    print("1st eigen value: %.2f%%" % (model.eigenvalues[0] / model.eigenvalues.sum()*100))
    print("2nd eigen value: %.2f%%" % (model.eigenvalues[1] / model.eigenvalues.sum()*100))
    print("variance kept with 10 components: %.2f%%" % (model.eigenvalues[:10].sum() / model.eigenvalues.sum()*100))

    X_transformed = model.transform(X)
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.4, c=y)
    ax.set_title(title)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Digit Label")

if __name__ == "__main__":
    N = 5000
    X, y = sklearn.datasets.fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    X = X[:N] / 255.
    y = y[:N]
    y = np.array(y, dtype=int)

    plt.figure(figsize=(19.2, 10.8))

    ax = plt.subplot(3,1,1)
    sep = "#"*10
    print("%s Linear kernel PCA %s" % (sep, sep))
    dim_red(X=X, y=y, n_components=2, kernel=pca.linear_kernel, ax=ax, title="Kernel PCA with linear kernel on MNIST")
    print(sep); print()
    ax = plt.subplot(3,1,2)
    print("%s RBF kernel PCA %s" % (sep, sep))
    print("Magnitude of good sigma:", pca.find_sigma(X))
    sigma = 4.8
    print("sigma =", sigma)
    dim_red(X=X, y=y, n_components=2, kernel=pca.radial_based_function(sigma), ax=ax, title="Kernel PCA with RBF kernel on MNIST")
    ax = plt.subplot(3,1,3)
    sigma = 10
    print("sigma =", sigma)
    dim_red(X=X, y=y, n_components=2, kernel=pca.radial_based_function(sigma), ax=ax, title="Kernel PCA with RBF kernel on MNIST")
    print(sep)

    plt.show()
