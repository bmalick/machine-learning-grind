import numpy as np

class QuadraticDiscriminantAnalysis:
    def __init__(self):
        self.priors = None
        self.means = None
        self.covs = None

    def __str__(self) -> str: return "QuadraticDiscriminantAnalysis"
    def __repr__(self) -> str: return "QuadraticDiscriminantAnalysis"

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.classes = list(set(y))
        self.num_classes = len(self.classes)

        self.priors = np.zeros(self.num_classes)
        self.means = np.zeros((self.num_classes, n_features))
        self.covs = np.zeros((self.num_classes, n_features, n_features))

        for c_idx, c in enumerate(self.classes):
            Nc = (y==c).sum()
            self.priors[c_idx] = Nc / n_samples

            xis = X[np.argwhere(y==c).reshape(-1), :]
            mean = xis.sum(axis=0) / Nc
            diff = (xis - mean)
            sigma = diff.T @ diff / Nc
            self.means[c_idx, :] = mean
            self.covs[c_idx, :] = sigma
        
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape)==1:
            probs = np.log(self.priors) # log trick

            for c_idx, c in enumerate(self.classes):
                diff = x - self.means[c_idx]
                cov = self.covs[c_idx]
                cov_inv = np.linalg.inv(cov)
                cov_det = np.linalg.det(cov)
                p = np.exp(- diff.T @ cov_inv @ diff / 2) / np.sqrt(2 * np.pi * cov_det)
                probs[c_idx] += np.log(p + 1e-10)

            probs -= probs.max() # max trick
            probs = np.exp(probs) # exp trick
            probs /= probs.sum() # sum trick

            return probs
        else: return np.array([self.predict_proba(xi) for xi in x])


    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=-1)

class LinearDiscriminantAnalysis(QuadraticDiscriminantAnalysis):
    def __init__(self):
        super().__init__()

    def __str__(self) -> str: return "LinearDiscriminantAnalysis"
    def __repr__(self) -> str: return "LinearDiscriminantAnalysis"

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.classes = list(set(y))
        self.num_classes = len(self.classes)

        self.priors = np.zeros(self.num_classes)
        self.means = np.zeros((self.num_classes, n_features))
        self.cov = np.zeros((n_features, n_features))

        for c_idx, c in enumerate(self.classes):
            Nc = (y==c).sum()
            self.priors[c_idx] = Nc / n_samples

            xis = X[np.argwhere(y==c).reshape(-1), :]
            mean = xis.sum(axis=0) / Nc
            diff = (xis - mean)
            sigma = diff.T @ diff
            self.means[c_idx, :] = mean
            self.cov += sigma
        
        self.cov /= n_samples
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape)==1:
            probs = np.log(self.priors) # log trick

            for c_idx, c in enumerate(self.classes):
                diff = x - self.means[c_idx]
                cov = self.cov
                cov_inv = np.linalg.inv(cov)
                cov_det = np.linalg.det(cov)
                p = np.exp(- diff.T @ cov_inv @ diff / 2) / np.sqrt(2 * np.pi * cov_det)
                probs[c_idx] += np.log(p + 1e-10)

            probs -= probs.max() # max trick
            probs = np.exp(probs) # exp trick
            probs /= probs.sum() # sum trick

            return probs
        else: return np.array([self.predict_proba(xi) for xi in x])

# TODO
# class DiagonalLDA
