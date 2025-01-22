import numpy as np
import pandas as pd


def mle_gaussian(x):
    """Maximum likelihood estimation for Gaussian distribution"""
    if isinstance(x, list): x = np.array(x)
    mu = x.mean()
    sigma = np.sqrt(((x-mu)**2).mean())
    return mu, max(sigma, 1e-10)

def mle_bernoulli(x):
    """Maximum likelihood estimation for Bernoulli distribution"""
    if isinstance(x, list): x = np.array(x)
    return np.mean(x)

def accuracy(y_pred, y_true):
    return (y_pred==y_true).mean()

def cross_validation(model, X, y, cv):
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
    def __init__(self, distribution: str):
        assert distribution in ["gaussian", "bernoulli"]
        self.distribution = distribution

    def get_py(self, y):
        """Calculate prior probabilities"""
        self.labels = np.unique(y).tolist()
        self.py = {l: (y==l).sum()/len(y) for l in self.labels}

    def estimate_distribution_params(self, X, y):
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

    def fit(self, X, y):
        self.get_py(y)
        self.estimate_distribution_params(X, y)
        return self

    def get_proba(self, xi, yi, feature_idx):
        """Calculate P(x_i|y) for a single feature value"""
        if self.distribution == "gaussian":
            mu, sigma = self.theta[yi][0][feature_idx], self.theta[yi][1][feature_idx]
            return np.exp(-(xi-mu)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi) * sigma)
        elif self.distribution == "bernoulli":
            p = self.theta[yi][feature_idx]
            return p if xi==1 else 1-p
        else: raise ValueError("This distribution is not taken account.")
    
    def predict_proba(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        probas = []
        for x in X:
            # Calculate log probabilities to avoid numerical underflow
            log_probs = []
            for label in self.labels:
                log_prob = np.log(self.py[label])
                for feature_idx, xi in enumerate(x):
                    prob = self.get_proba(xi, label, feature_idx)
                    # Add small epsilon to avoid log(0)
                    log_prob += np.log(max(prob, 1e-10))
                log_probs.append(log_prob)
                
            # Convert log probabilities back to probabilities
            log_probs = np.array(log_probs)
            probs = np.exp(log_probs - np.max(log_probs)) # max normalization that helps prevent numerical underflow/overflow when working with probabilities
            probs = probs / probs.sum()
            probas.append(probs)
            
        return np.array(probas)
    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)


