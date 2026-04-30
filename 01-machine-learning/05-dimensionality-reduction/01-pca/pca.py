import numpy as np
import sklearn.metrics

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> None:
        self.centered = np.mean(X, axis=0)
        # cov = np.dot((X-self.centered).T, X-self.centered)
        cov = np.dot((X-self.centered).T, X-self.centered) / X.shape[0]
        # cov = np.dot((X-self.centered).T, X-self.centered) / (X.shape[0]-1)
        # self.eigenvalues, self.eigenvectores = np.linalg.eig(cov)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(cov)
        # eigh is for Hermitian/symmetric matrices
        # eig is for general matrices
        # eigh is faster and more stable for Hermitian/symmetric matrices
        # eig is more general but slower for symmetric matrices compared to eigh
        # eigh returns sorted eigenvalues, while eig does not guarantee any order
        self.eigenvectors = self.eigenvectors[:, ::-1]
        self.eigenvalues = self.eigenvalues[::-1]

        self.variance_ratio = self.eigenvalues[:self.n_components] / self.eigenvalues.sum()

    def transform(self, X: np.array) -> np.ndarray:
        return np.dot(X-self.centered, self.eigenvectors[:, :self.n_components])

    def fit_transform(self, X: np.array) -> np.ndarray:
        self.fit(X)
        return self.transform(X)



def linear_kernel(x: np.ndarray, y: np.ndarray) -> float:
    return np.dot(x.T, y)

def get_sigma(gamma: float) -> float: return 1/np.sqrt(gamma*2)

def get_gamma(sigma: float) -> float: return 1/sigma**2

def find_sigma(X: np.ndarray) -> float: return sklearn.metrics.euclidean_distances(X).mean()

def radial_based_function(sigma: float):
    def k(x: np.ndarray, y: np.ndarray) -> float:
        diff = x-y
        return np.exp(-np.dot(diff, diff)/2/sigma**2)
    return k

def polynomial_kernel(gamma: float, r: float, d: int):
    def k(x: np.ndarray, y: np.ndarray) -> float:
        return (gamma*np.dot(x.T, y) + r)**d
    return k

def sigmoid_kernel(gamma: float, r: float):
    def k(x: np.ndarray, y: np.ndarray) -> float:
        return np.tanh(gamma*np.dot(x.T, y) + r)
    return k


class kPCA:
    def __init__(self, n_components: int, kernel):
        self.n_components = n_components
        self.kernel = kernel

    def center(self, G: np.ndarray) -> np.ndarray:
        N = G.shape[0]
        mat_center = np.identity(N) - np.ones((N,N)) / N
        return mat_center@G@mat_center
    
    def gram(self, X: np.ndarray) -> np.ndarray:
        return np.array([[self.kernel(xi,xj) for xi in X] for xj in X])

    def fit(self, X: np.ndarray) -> None:
        K = self.gram(X)
        K = self.center(K)
        cov = np.dot(K.T, K) / K.shape[0]
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(cov)
        self.eigenvectors = self.eigenvectors[:, ::-1]
        self.eigenvalues = self.eigenvalues[::-1]
        self.variance_ratio = self.eigenvalues[:self.n_components] / self.eigenvalues.sum()

    def transform(self, X: np.array) -> np.ndarray:
        G = self.gram(X)
        return np.dot(G, self.eigenvectors[:, :self.n_components])

    def fit_transform(self, X: np.array) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
