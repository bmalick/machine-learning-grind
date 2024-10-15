import numpy as np

class LeastSquares:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_biased = np.c_[np.ones(len(X)), X]
        self.beta_hat = np.linalg.inv(X_biased.T@X_biased)@X_biased.T@y
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        x_biased = np.c_[np.ones(len(x)), x]
        return x_biased@self.beta_hat

