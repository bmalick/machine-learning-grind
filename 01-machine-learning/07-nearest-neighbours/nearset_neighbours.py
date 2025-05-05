import numpy as np

class NearestNeighbours:
    def __init__(self, k: int = None, gaussian_kernel: bool = False, sigma: float = None):
        assert k is None or sigma is None
        self.k = k
        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sqrt(((x-y)**2).sum(axis=-1))
    
    def gaussian_weight(self, xi: np.ndarray, x: np.ndarray) -> float:
        return np.exp(-self.distance(xi, x)**2 / 2 / self.sigma**2)

    def gaussian_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.gaussian_weight(x, xi) for xi in self.X])

    def predict(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape)==1:
            if self.gaussian_kernel:
                W = self.gaussian_matrix(x)
                return W@self.y / W.sum()
            else:
                indexes = self.k_nearest(x)
                return self.y[indexes].sum(axis=-1) / self.k
        else: return np.array([self.predict(xi) for xi in x])

    def k_nearest(self, x: np.ndarray) -> np.ndarray:
        distances = {i: self.distance(x,xi) for i,xi in enumerate(self.X)}
        indexes = sorted(distances.items(), key=lambda x: x[1])
        indexes = [x[0] for x in indexes[:self.k]]
        return np.array(indexes)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

