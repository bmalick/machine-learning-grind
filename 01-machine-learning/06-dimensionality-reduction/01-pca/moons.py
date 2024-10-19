import matplotlib.pyplot as plt
import sklearn.datasets
import pca

from sklearn.decomposition import PCA

if __name__ ==  "__main__":
    N = 100
    X, y = sklearn.datasets.make_moons(N, random_state=42)

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
    print("variance ratio: ", model.variance_ratio.tolist())


    ax = plt.subplot(1,2,1)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.4)
    # ax.plot([w0[0],w0[0]+w1[0]], [w0[1], w0[1]+w1[1]], color="red")
    # ax.plot([w0[0],w0[0]+w2[0]], [w0[1], w0[1]+w2[1]], color="green")
    ax.set_title("Principal components")

    ax = plt.subplot(1,2,2)
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.4)
    ax.set_title("Transformed")

    # plt.show()
    mpca = PCA(n_components=2)
    mpca.fit(X)
    print(mpca.explained_variance_ratio_)
    print(mpca.components_.T[:, 0])
    print(mpca.components_.T[:, 1])
