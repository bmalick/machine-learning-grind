import numpy as np
from decision_tree import Node

def gini(data):
    if isinstance(data, Node):
        return gini(data.samples)
    _, y = data
    if len(y) == 0:
        return 0
    probas = {yi: 0 for yi in set(y)}
    for yi in y:
        probas[yi] += 1
    probas = {k: v / len(y) for k, v in probas.items()}
    return 1 - sum([pi ** 2 for pi in probas.values()])

def accuracy(y_pred: np.ndarray, y_true: np.ndarray):
    return (y_pred==y_true).mean()

def empirical_risk(classificer, X, y):
    y_pred = classificer.predict(X)
    return (y!=y_pred).sum() / len(y)

def mse(data):
    if isinstance(data, Node):
        return gini(data.samples)
    _, y = data
    if len(y) == 0:
        return 0
    return ((y-y.mean(axis=-1))**2).mean()
    
# TODO
def entropy(data): pass

# TODO
def misclassification_rate(data): pass

def create_dataset(n=500, noise=0.5):
    X = np.random.normal(loc=0, scale=1, size=(n, 1))
    y = X**2 + noise*np.random.normal(loc=0, scale=1, size=(n, 1))
    X = (X-X.min()) / (X.max() - X.min())
    y = (y-y.min()) / (y.max() - y.min())
    return X, y.reshape(-1)