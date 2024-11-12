
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import pca

def get_dataset(name: str, N: int):
    if name=="digits":
        X, y = sklearn.datasets.fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X = X[:N] / 255.
        y = y[:N]
        y = np.array(y, dtype=int)
    elif name=="moons":
        X, y = sklearn.datasets.make_moons(N, random_state=42)
        y = np.array(y, dtype=int)
    elif name=="swiss_roll":
        X, y = sklearn.datasets.make_swiss_roll(n_samples=N, noise=0.1, random_state=42)
        
    return X, y

def model_display(model, X, y, ax, title):
    print("2 first eigen values = ", model.eigenvalues.tolist()[:2])
    print("1st eigen value: %.2f%%" % (model.eigenvalues[0] / model.eigenvalues.sum()*100))
    print("2nd eigen value: %.2f%%" % (model.eigenvalues[1] / model.eigenvalues.sum()*100))
    print("variance ratio: ", model.variance_ratio.tolist())
    try:
        print("variance kept with 10 components: %.2f%%" % (model.eigenvalues[:10].sum() / model.eigenvalues.sum()*100))
    except : pass
    X_transformed = model.transform(X)
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.4, c=y)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(title)
    # ax.colorbar(label="Digit Label")

def dim_red(X, y, n_components, kernel, ax, title):
    model = pca.kPCA(n_components=n_components, kernel=kernel)
    model.fit(X)
    model_display(model, X, y, ax, title)
