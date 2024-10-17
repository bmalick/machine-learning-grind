import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import pca

if __name__ == "__main__":
    N = 5000
    X, y = sklearn.datasets.fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    X = X[:N] / 255.
    y = y[:N]
    y = np.array(y, dtype=int)

    n_components = 2
    model = pca.PCA(n_components=n_components)
    model.fit(X)

    X_transformed = model.transform(X)
    print("2 first eigen values = ", model.eigenvalues.tolist()[:2])
    w0 = model.centered
    w1 = model.eigenvectors[:, 0]
    w2 = model.eigenvectors[:, 1]
    print("1st eigen value: %.2f%%" % (model.eigenvalues[0] / model.eigenvalues.sum()*100))
    print("2nd eigen value: %.2f%%" % (model.eigenvalues[1] / model.eigenvalues.sum()*100))
    print("variance kept with 10 components: %.2f%%" % (model.eigenvalues[:10].sum() / model.eigenvalues.sum()*100))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.4, c=y)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of MNIST")
    plt.colorbar(label="Digit Label")
    plt.show()
