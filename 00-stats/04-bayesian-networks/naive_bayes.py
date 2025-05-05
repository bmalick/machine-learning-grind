import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

# TODO: add message length feature; view distribution and estimate it with poisson distribution

def mle_gaussian(x: np.ndarray) -> Tuple[float, float]:
    """Maximum likelihood estimation for Gaussian distribution"""
    if isinstance(x, list): x = np.array(x)
    mu = x.mean()
    sigma = np.sqrt(((x-mu)**2).mean())
    return mu, max(sigma, 1e-10)

def mle_bernoulli(x: np.ndarray) -> float:
    """Maximum likelihood estimation for Bernoulli distribution"""
    if isinstance(x, list): x = np.array(x)
    return np.mean(x)

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


class NaiveBayesFromScratch:
    def __init__(self, distribution: str = "bernoulli"):
        assert distribution in ["gaussian", "bernoulli"]
        self.distribution = distribution

    def get_py(self, y: np.ndarray) -> None:
        """Calculate prior probabilities"""
        self.labels = np.unique(y).tolist()
        self.py = {l: (y==l).sum()/len(y) for l in self.labels}

    def estimate_distribution_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """Estimate distribution parameters for each feature and class"""
        mle_estimator = mle_bernoulli if self.distribution=="bernoulli" else mle_gaussian
        self.theta = {}
        for label in self.labels:
            mask = (y == label)
            X_class = X[mask]
            
            if self.distribution == "gaussian":
                # Calculate parameters for each feature
                params = [mle_estimator(X_class[:, i]) for i in range(X.shape[1])]
                self.theta[label] = list(zip(*params))
            else:
                self.theta[label] = [mle_estimator(X_class[:, i]) for i in range(X.shape[1])]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.get_py(y)
        self.estimate_distribution_params(X, y)
        return self

    def get_proba(self, xi: float, yi: int, feature_idx: int) -> float:
        """Calculate P(x_i|y) for a single feature value"""
        if self.distribution == "gaussian":
            mu, sigma = self.theta[yi][0][feature_idx], self.theta[yi][1][feature_idx]
            return np.exp(-(xi-mu)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma)
        elif self.distribution == "bernoulli":
            p = self.theta[yi][feature_idx]
            return p if xi==1 else 1-p
        else: raise ValueError("This distribution is not taken account.")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        n_samples = X.shape[0]
        n_classes = len(self.labels)
        log_probas = np.zeros((n_samples, n_classes))

        # Calculate log probabilities to avoid numerical underflow
        log_probas += np.log([self.py[label] for label in self.labels])

        for feature_idx in range(X.shape[1]):
            feature_probas = np.array([[self.get_proba(x[feature_idx], label, feature_idx)
                                        for label in self.labels]
                                        for x in X])
            # Add small epsilon to avoid log(0)
            log_probas += np.log(np.maximum(feature_probas, 1e-10))
                
        # Convert log probabilities back to probabilities
        probas = np.exp(log_probas - np.max(log_probas, axis=1, keepdims=True)) # max normalization that helps prevent numerical underflow/overflow when working with probabilities
        probas = probas / probas.sum(axis=1, keepdims=True)
        return probas

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

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
