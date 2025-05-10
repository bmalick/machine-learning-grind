#!/home/malick/miniconda3/envs/pt/bin/python3

import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import utils
import gda

if __name__=="__main__":
    X, y = utils.get_iris()

    X = (X-X.mean(axis=0)) / X.std(axis=0)
    X = PCA(n_components=2).fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    qda_model = gda.QuadraticDiscriminantAnalysis()
    qda_model.fit(X_train, y_train)

    lda_model = gda.LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)

    xbounds = (X[:,0].min(), X[:,0].max())
    ybounds = (X[:,1].min(), X[:,1].max())

    colors = ['#ff0000', '#00ff00', '#0000ff']

    fig, ax = plt.subplots(figsize=(10,8))

    utils.decision_boundary(ax=ax, xbounds=xbounds, ybounds=ybounds,
                      colors=colors, model=qda_model, N=300, margin=0.5)
    ax.scatter(X[:, 0], X[:, 1], c=[colors[yi] for yi in  y], s=60, alpha=0.5)
    for i in range(3):
        utils.plot_ellipsoid(ax=ax, mu=qda_model.means[i], sigma=qda_model.covs[i], color=colors[i])
    ax.set_title("Decision boundary for GDA")
    os.makedirs("images", exist_ok=True)
    fig.savefig("images/decision-boundary-gda.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(10,8))

    utils.decision_boundary(ax=ax, xbounds=xbounds, ybounds=ybounds,
                      colors=colors, model=lda_model, N=300, margin=0.5)
    ax.scatter(X[:, 0], X[:, 1], c=[colors[yi] for yi in  y], s=60, alpha=0.5)
    for i in range(3):
        utils.plot_ellipsoid(ax=ax, mu=lda_model.means[i], sigma=lda_model.cov, color=colors[i])
    ax.set_title("Decision boundary for LDA")
    fig.savefig("images/decision-boundary-lda.png")
    plt.show()

    for m in [qda_model, lda_model]:
        print(f"Accuracy for {m}:", utils.accuracy(m, X_test, y_test))
        print(f"Log loss for {m}:", utils.logloss(m, X_test, y_test))
