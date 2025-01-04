#!/home/malick/miniconda3/envs/pt/bin/python
import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets


import pca
import utils

def linear_example():
    N = 500
    mu = np.array([0.5, 1.0])
    sigma = np.array([[0.33, 0.46], [0.33, 0.76]])
    X = np.random.multivariate_normal(mean=mu, cov=sigma, size=N)

    n_components = 2
    model = pca.PCA(n_components=n_components)
    model.fit(X)
    X_transformed = model.transform(X)
    print("eigen values = ", model.eigenvalues.tolist())
    w0 = model.centered
    w1 = model.eigenvectors[:, 0]
    w2 = model.eigenvectors[:, 1]
    print("w0 = ", w0)
    print("w1 = ", w1)
    print("w2 = ", w2)
    print("1st eigen value: %.2f%%" % (model.eigenvalues[0] / model.eigenvalues.sum()*100))
    print("2nd eigen value: %.2f%%" % (model.eigenvalues[1] / model.eigenvalues.sum()*100))

    ax = plt.subplot(1,2,1)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.4)
    ax.plot([w0[0],w0[0]+w1[0]], [w0[1], w0[1]+w1[1]], color="red")
    ax.plot([w0[0],w0[0]+w2[0]], [w0[1], w0[1]+w2[1]], color="green")
    ax.set_title("Principal components")

    ax = plt.subplot(1,2,2)
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.4)
    ax.set_title("Transformed")

    plt.show()


def digits():
    N = 5000
    X, y = utils.get_dataset(name="digits", N=N)
    n_components = 2
    model = pca.PCA(n_components=n_components)
    model.fit(X)

    fig, ax = plt.subplots()
    utils.model_display(model, X, y, ax, "PCA of MNIST")
    # plt.colorbar(label="Digit Label")
    plt.show()


def digits_kpca():
    N = 5000
    X, y = utils.get_dataset(name="digits", N=N)

    plt.figure(figsize=(19.2, 10.8))

    ax = plt.subplot(3,1,1)
    sep = "#"*10
    print("%s Linear kernel PCA %s" % (sep, sep))
    utils.dim_red(X=X, y=y, n_components=2, kernel=pca.linear_kernel, ax=ax, title="Kernel PCA with linear kernel on MNIST")
    print(sep); print()
    ax = plt.subplot(3,1,2)
    print("%s RBF kernel PCA %s" % (sep, sep))
    print("Magnitude of good sigma:", pca.find_sigma(X))
    sigma = 4.8
    print("sigma =", sigma)
    utils.dim_red(X=X, y=y, n_components=2, kernel=pca.radial_based_function(sigma), ax=ax, title="Kernel PCA with RBF kernel on MNIST")
    ax = plt.subplot(3,1,3)
    sigma = 10
    print("sigma =", sigma)
    utils.dim_red(X=X, y=y, n_components=2, kernel=pca.radial_based_function(sigma), ax=ax, title="Kernel PCA with RBF kernel on MNIST")
    print(sep)

    plt.show()

def moons():
    N = 100
    X, y = utils.get_dataset(name="moons", N=N)

    n_components = 2
    model = pca.PCA(n_components=n_components)
    model.fit(X)

    plt.figure(figsize=(12,6))

    ax = plt.subplot(1,2,1)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.4)
    ax.set_title("Principal components")

    ax = plt.subplot(1,2,2)
    utils.model_display(model, X, y, ax, "Transformed")
    plt.show()


def moons_kpca():
    N = 100
    X, y = utils.get_dataset(name="moons", N=N)

    plt.figure(figsize=(19.2, 10.8))

    ax = plt.subplot(2,2,1)
    sep = "#"*10
    print("%s Linear kernel PCA %s" % (sep, sep))
    utils.dim_red(X=X, y=y, n_components=2, kernel=pca.linear_kernel, ax=ax, title="Kernel PCA with linear kernel on moons")
    print(sep); print()

    ax = plt.subplot(2,2,2)
    print("%s RBF kernel PCA %s" % (sep, sep))
    sigma = pca.get_sigma(15)
    utils.dim_red(X=X, y=y, n_components=2, kernel=pca.radial_based_function(sigma), ax=ax, title="Kernel PCA with RBF kernel on moons")
    print(sep); print()

    ax = plt.subplot(2,2,3)
    print("%s Polynomial kernel PCA %s" % (sep, sep))
    gamma = 0.2
    r = -1
    d = 2
    utils.dim_red(X=X, y=y, n_components=2, kernel=pca.polynomial_kernel(gamma,r,d), ax=ax, title="Kernel PCA with polynomial kernel on moons")
    print(sep); print()

    ax = plt.subplot(2,2,4)
    print("%s Sigmoid kernel PCA %s" % (sep, sep))
    gamma = 0.1
    r = 1
    utils.dim_red(X=X, y=y, n_components=2, kernel=pca.sigmoid_kernel(gamma,r), ax=ax, title="Kernel PCA with sigmoid kernel on moons")
    print(sep); print()

    plt.show()

def swiss_roll():
    N = 1000
    X, y = utils.get_dataset(name="swiss_roll", N=N)

    n_components = 2
    model = pca.PCA(n_components=n_components)
    model.fit(X)

    fig, ax = plt.subplots()
    utils.model_display(model, X, y, ax, "PCA of Swiss Roll")
    # plt.colorbar(label="Digit Label")
    plt.show()


def swiss_roll_pca():
    N = 1000
    X, y = sklearn.datasets.make_swiss_roll(n_samples=N, noise=0.1, random_state=42)

    plt.figure(figsize=(19.8, 10.8))

    ax = plt.subplot(2,2,1)
    sep = "#"*10
    print("%s Linear kernel PCA %s" % (sep, sep))
    utils.dim_red(X=X, y=y, n_components=2, kernel=pca.linear_kernel, ax=ax, title="Kernel PCA with linear kernel on Swiss Roll")
    print(sep); print()
    ax = plt.subplot(2,2,2)
    print("%s RBF kernel PCA %s" % (sep, sep))
    print("Magnitude of good sigma:", pca.find_sigma(X))
    sigma = 3.5
    print("sigma =", sigma)
    utils.dim_red(X=X, y=y, n_components=2, kernel=pca.radial_based_function(sigma), ax=ax, title="Kernel PCA with RBF kernel on Swiss Roll")
    ax = plt.subplot(2,2,3)
    sigma = 15
    print("sigma =", sigma)
    utils.dim_red(X=X, y=y, n_components=2, kernel=pca.radial_based_function(sigma), ax=ax, title="Kernel PCA with RBF kernel on Swiss Roll")
    print(sep)
    ax = plt.subplot(2,2,4)
    print("%s Sigmoid kernel PCA %s" % (sep, sep))
    sigma = 22
    print("sigma =", sigma)
    gamma = pca.get_gamma(sigma)
    # gamma = 1e-3
    r = 1
    utils.dim_red(X=X, y=y, n_components=2, kernel=pca.sigmoid_kernel(gamma=gamma,r=r), ax=ax, title="Kernel PCA with Sigmoid kernel on Swiss Roll")
    print(sep)

    plt.show()

if __name__ == "__main__":
    functions = [
        linear_example,
        digits, digits_kpca,
        moons, moons_kpca,
        swiss_roll, swiss_roll_pca
    ]
    if len(sys.argv) !=2:
        print("Usage: %s <function id>" % sys.argv[0])
        print()
        print("id | function")
        print("---+"+'-'*20)
        for id, f in enumerate(functions):
            print("%d  | %s" %(id, f.__name__))
        sys.exit()

    id = int(sys.argv[1])
    if(id < 0 or id >= len(functions)) :
        print("Function id %d is invalid (should be in [0, %d])" % (id, len(functions)-1))
        sys.exit()
    functions[id]()
