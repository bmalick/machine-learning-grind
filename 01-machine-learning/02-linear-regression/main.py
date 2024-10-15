import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import utils
from linear_model import LeastSquares

def least_squares(a, b, ax, colors = np.array(['#ff7f00', '#377eb8'])):
    X, y = utils.create_dataset(a=1.9, b=-0.14)
    model = LeastSquares()
    model.fit(X, y)

    ax.scatter(X, y, s=5)
    bounds = np.array([X.min(), X.max()])
    ax.plot(bounds, model.predict(bounds), c="orange", linestyle="--")
    title = r"Least Squares - (a = %.3f, b = %.3f) - $\hat{\beta} = (%.3f, %.3f)$" % (a, b, model.beta_hat[1], model.beta_hat[0])
    print(title + " - custom data")
    ax.set_title(title)


def main():
    figure, axes = plt.subplots(2,2,figsize=(20,15))
    axes = axes.ravel()

    least_squares(a=1.9, b=-0.14, ax=axes[0])
    least_squares(a=1.9, b=-0.14, ax=axes[1])
    least_squares(a=1.9, b=-0.14, ax=axes[2])
    least_squares(a=1.9, b=-0.14, ax=axes[3])
    plt.show()

if __name__ == "__main__": main()
