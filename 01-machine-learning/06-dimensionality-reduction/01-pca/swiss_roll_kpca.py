import matplotlib.pyplot as plt
import sklearn.datasets
import pca

def dim_red(X, y, n_components, kernel, ax, title):
    model = pca.kPCA(n_components=n_components, kernel=kernel)
    model.fit(X)

    print("2 first eigen values = ", model.eigenvalues.tolist()[:2])
    print("1st eigen value: %.2f%%" % (model.eigenvalues[0] / model.eigenvalues.sum()*100))
    print("2nd eigen value: %.2f%%" % (model.eigenvalues[1] / model.eigenvalues.sum()*100))
    print("variance ratio: ", model.variance_ratio.tolist())

    X_transformed = model.transform(X)
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.4, c=y)
    ax.set_title(title)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Digit Label")

if __name__ == "__main__":
    N = 1000
    X, y = sklearn.datasets.make_swiss_roll(n_samples=N, noise=0.1, random_state=42)

    plt.figure(figsize=(19.8, 10.8))

    ax = plt.subplot(2,2,1)
    sep = "#"*10
    print("%s Linear kernel PCA %s" % (sep, sep))
    dim_red(X=X, y=y, n_components=2, kernel=pca.linear_kernel, ax=ax, title="Kernel PCA with linear kernel on Swiss Roll")
    print(sep); print()
    ax = plt.subplot(2,2,2)
    print("%s RBF kernel PCA %s" % (sep, sep))
    print("Magnitude of good sigma:", pca.find_sigma(X))
    sigma = 3.5
    print("sigma =", sigma)
    dim_red(X=X, y=y, n_components=2, kernel=pca.radial_based_function(sigma), ax=ax, title="Kernel PCA with RBF kernel on Swiss Roll")
    ax = plt.subplot(2,2,3)
    sigma = 15
    print("sigma =", sigma)
    dim_red(X=X, y=y, n_components=2, kernel=pca.radial_based_function(sigma), ax=ax, title="Kernel PCA with RBF kernel on Swiss Roll")
    print(sep)
    ax = plt.subplot(2,2,4)
    print("%s Sigmoid kernel PCA %s" % (sep, sep))
    sigma = 22
    print("sigma =", sigma)
    gamma = pca.get_gamma(sigma)
    # gamma = 1e-3
    r = 1
    dim_red(X=X, y=y, n_components=2, kernel=pca.sigmoid_kernel(gamma=gamma,r=r), ax=ax, title="Kernel PCA with Sigmoid kernel on Swiss Roll")
    print(sep)

    plt.show()
