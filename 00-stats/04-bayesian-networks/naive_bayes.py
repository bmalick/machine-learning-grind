import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

# TODO: add message length feature; view distribution and estimate it with poisson distribution

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return (y_pred==y_true).mean()

def cross_validation(model, X: np.ndarray, y: np.ndarray, cv: int) -> np.ndarray:
    split = len(X) // cv
    X = X[:cv*split]
    y = y[:cv*split]
    scores = []
    for i in range(cv):
        test_idx = [n for n in range(i*split, (i+1)*split)]
        train_idx = [n for n in range(0, i*split) if n>-1] + [n for n in range((i+1)*split, cv*split) if n>-1]
        X_test, y_test = X[test_idx], y[test_idx]
        X_train, y_train = X[train_idx], y[train_idx]
        model.fit(X_train, y_train)
        scores.append(accuracy(y_pred=model.predict(X_test), y_true=y_test))
    return np.array(scores)

class CategoricalNaiveBayes:
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, feature_dim = X.shape
        self.classes = list(set(y))
        self.features = set()
        for i in range(n_samples):
            for xi in X[i]:
                self.features.add(xi)

        self.num_features = len(self.features)
        self.num_classes = len(self.classes)
        self.feature_dim = feature_dim

        self.class_priors = np.array([(y==c).sum() / n_samples for c in self.classes]) # pi_c
        self.mles = np.zeros((feature_dim, self.num_classes, self.num_features)) # theta_jc

        for j in range(feature_dim):
            for c in range(len(self.classes)):
                Nc = (y==self.classes[c]).sum()
                for v in range(self.num_features):
                    Njcv = ((X[:, j]==v) & (y==self.classes[c])).sum()
                    self.mles[j][c][v] = Njcv / Nc

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape)==1:
            probs = np.log(self.class_priors) # log trick

            for c_idx, c in enumerate(self.classes):
                for j in range(len(x)):
                    xj = x[j]
                    # probs[c_idx][j] = self.mles[j][c_idx][xj]
                    probs[c_idx] += np.log(self.mles[j][c_idx][xj] + 1.e-10) # log trick
            
            # probs = self.class_priors * np.prod(probs, axis=1)
            # return print(probs / probs.sum())

            probs = probs - np.max(probs) # max trick
            probs = np.exp(probs) # exp trick
            probs /= probs.sum() # sum trick
            return probs
        else:
            return np.array([self.predict_proba(xi) for xi in x])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=-1)

class GaussianNaiveBayes:
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, feature_dim = X.shape
        self.classes = list(set(y))
        self.features = set()
        for i in range(n_samples):
            for xi in X[i]:
                self.features.add(xi)

        self.num_features = len(self.features)
        self.num_classes = len(self.classes)
        self.feature_dim = feature_dim

        self.class_priors = np.array([(y==c).sum() / n_samples for c in self.classes]) # pi_c
        self.mles = np.zeros((feature_dim, self.num_classes, 2)) # theta_jc

        for j in range(feature_dim):
            for c in range(len(self.classes)):
                Nc = (y==self.classes[c]).sum()
                theta = X[np.argwhere(y==self.classes[c]), j].sum() / Nc
                sigma_square = (X[np.argwhere(y==self.classes[c]), j].sum() - theta)**2 / Nc
                self.mles[j][c][0] = theta
                self.mles[j][c][1] = sigma_square + 1e-10

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape)==1:
            probs = np.log(self.class_priors) # log trick

            for c_idx, c in enumerate(self.classes):
                for j in range(len(x)):
                    xj = x[j]
                    mu = self.mles[j][c_idx][0]
                    sigma_square = self.mles[j][c_idx][1]
                    p = np.exp(-(xj-mu)**2 / (2*sigma_square)) / (np.sqrt(2*np.pi*sigma_square))
                    # probs[c_idx][j] = p
                    probs[c_idx] += np.log(p + 1e-10)
            
            # probs = self.class_priors * np.prod(probs, axis=1)
            # return print(probs / probs.sum())

            probs = probs - np.max(probs) # max trick
            probs = np.exp(probs) # exp trick
            probs /= probs.sum() # sum trick
            return probs
        else:
            return np.array([self.predict_proba(xi) for xi in x])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=-1)

def calibration_curve_from_scratch(y_true: np.ndarray, pred_probas: np.ndarray, k: int=10):
    intervals = np.linspace(0, 1, k+1)
    calib =  []
    for i in range(1,len(intervals)):
        a, b = intervals[i-1], intervals[i]
        p = pred_probas[pred_probas[:, 1]>=a]
        p = p[p[:, 1]<=b]
        n1 = y_true[(pred_probas[:, 1]>=a) & (pred_probas[:, 1]<=b)]
        if len(p)!=0:
            calib.append([p[:, 1].mean(), (n1==1).sum() / len(p)])
    calib = np.array(calib)
    plt.plot(calib[:, 0], calib[:, 1], "-y", label="Real calibration")
    plt.plot([0, 1], [0, 1], "--b", label="Expected calibration")
    plt.legend(); plt.show()
