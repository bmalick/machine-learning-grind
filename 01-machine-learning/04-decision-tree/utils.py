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

# TODO
def entropy(data): pass

# TODO
def misclassification_rate(data): pass

