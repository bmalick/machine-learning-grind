#!/home/malick/miniconda3/envs/pt/bin/python
import sys
import utils
import svm
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# mean = X.mean(axis=0)
# std = X.std(axis=0)
# X = (X - mean) / std



# Linear SVC
def linear_svc_iris(name):
    X, y, xbounds, ybounds = utils.get_data(name=name)

    fig, axes = plt.subplots(2,2, figsize=(19.2,10.8))
    axes = axes.ravel()
    for i, c_param in enumerate([0.01, 0.1, 1, 100]):
        model = SVC(C=c_param, kernel="linear")
        model.fit(X, y)
        print("#########")
        print("C=%.2f" % c_param)
        print("#########")
        utils.apply_cross_validation(model, X, y)
        utils.decision_boundary(axes[i], model, X, y, xbounds, ybounds, margin=0.1)
        axes[i].set_title(r"C=%.2f" % c_param)
    plt.show()

def linear_svc1(): linear_svc_iris("iris1")
def linear_svc2(): linear_svc_iris("iris2")


def moons1():
    X, y, xbounds, ybounds = utils.get_data(name="moons", n=100)

    fig, axes = plt.subplots(2,2, figsize=(19.2,10.8))
    axes = axes.ravel()
    model = SVC(C=1., kernel="linear")
    model.fit(X, y)
    print("#########")
    print("C=1.0")
    print("#########")
    utils.apply_cross_validation(model, X, y)
    utils.decision_boundary(axes[0], model, X, y, xbounds, ybounds, margin=0.1)
    axes[0].set_title(r"C=1.0")
    for i, d in enumerate([2, 3, 4]):
        model = SVC(C=1., kernel="poly", degree=d, coef0=1.)
        model.fit(X, y)
        print("#########")
        print("Polynomial kernel: C=1.0, d=%d, coef0=1." % d)
        print("#########")
        utils.apply_cross_validation(model, X, y)
        utils.decision_boundary(axes[i+1], model, X, y, xbounds, ybounds, margin=0.1)
        axes[i+1].set_title(r"C=1.0, d=%d, coef0=1." % d)
    plt.show()

def moons2():
    X, y, xbounds, ybounds = utils.get_data(name="moons", n=100)

    fig, axes = plt.subplots(2,2, figsize=(19.2,10.8))
    axes = axes.ravel()
    for i, c_param in enumerate([0.1, 1, 10, 100]):
        model = SVC(C=c_param, kernel="poly", degree=3, coef0=1.)
        model.fit(X, y)
        print("#########")
        print("Polynomial kernel: C=%.2f, d=3, coef0=1." % c_param)
        print("#########")
        utils.apply_cross_validation(model, X, y)
        utils.decision_boundary(axes[i], model, X, y, xbounds, ybounds, margin=0.1)
        axes[i].set_title(r"C=%.2f, d=3, coef0=1." % c_param)
    plt.show()

def moons3():
    X, y, xbounds, ybounds = utils.get_data(name="moons", n=100)
    print("#########")
    good_sigma = utils.find_sigma(X)
    print("good sigma: s%.2f (gamm=%.2f)" % (good_sigma, utils.get_gamma(good_sigma)))
    print("#########")

    fig, axes = plt.subplots(2,2, figsize=(19.2,10.8))
    axes = axes.ravel()
    def run(sigma, c_param, ax):
        model = SVC(C=c_param, kernel="rbf", gamma=utils.get_gamma(sigma))
        model.fit(X, y)
        print("#########")
        print("RBF kernel: C=%.2f, sigma=%.2f" % (c_param, sigma))
        print("#########")
        utils.apply_cross_validation(model, X, y)
        utils.decision_boundary(ax, model, X, y, xbounds, ybounds, margin=0.1)
        ax.set_title(r"C=%.2f, $\sigma$ =%.2f" % (c_param, sigma))

    run(sigma=0.1, c_param=1, ax=axes[0])
    run(sigma=0.1, c_param=10, ax=axes[1])
    run(sigma=good_sigma, c_param=10, ax=axes[2])
    run(sigma=good_sigma, c_param=1, ax=axes[3])
    plt.show()

def moons4():
    X, y, xbounds, ybounds = utils.get_data(name="moons", n=100)
    print("#########")
    good_sigma = utils.find_sigma(X)
    print("good sigma: s%.2f (gamm=%.2f)" % (good_sigma, utils.get_gamma(good_sigma)))
    print("#########")

    fig, axes = plt.subplots(2,2, figsize=(19.2,10.8))
    axes = axes.ravel()
    def run(sigma, c_param, ax):
        model = SVC(C=c_param, kernel="rbf", gamma=utils.get_gamma(sigma))
        model.fit(X, y)
        print("#########")
        print("RBF kernel: C=%.2f, sigma=%.2f" % (c_param, sigma))
        print("#########")
        utils.apply_cross_validation(model, X, y)
        utils.decision_boundary(ax, model, X, y, xbounds, ybounds, margin=0.1)
        ax.set_title(r"C=%.2f, $\sigma$ =%.2f" % (c_param, sigma))

    run(sigma=10, c_param=1, ax=axes[0])
    run(sigma=10, c_param=10, ax=axes[1])
    run(sigma=good_sigma, c_param=10, ax=axes[2])
    run(sigma=good_sigma, c_param=1, ax=axes[3])
    plt.show()

if __name__ == "__main__":
    functions = [
        linear_svc1, linear_svc2,
        moons1, moons2, moons3, moons4,
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
