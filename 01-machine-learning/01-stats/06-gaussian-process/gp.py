import numpy as np
import matplotlib.pyplot as plt


# Kernels
def squared_exp(sigma: float, l: float):
    return lambda x, y: sigma**2 * np.exp(-(x-y)**2 / (2*l**2))

def brownian():
    return lambda x,y: min(x,y)

class GaussianProcess:
    def __init__(self, mu, k):
        self.mu = np.vectorize(mu)
        self.k = np.vectorize(k)

    def sample(self, x: float, n: int=100):
        return np.random.normal(loc=self.mu(x), scale=self.k(x,x), size=n)

    def __call__(self, x):
        return self.sample(x, n=1)[0]

    def compute_mu(self, x): return self.mu(x)

    def compute_cov(self, x1, x2):
        return self.k(x1.reshape(-1,1), x2.reshape(1,-1))

    def represent_prior(self, num_samples: int=5, xmin: float=-5., xmax: float=5., n: int=100, ax=None, color=None):
        x = np.linspace(xmin,xmax,n)
        cov = self.compute_cov(x,x)
        mean = self.compute_mu(x)
        y = np.random.multivariate_normal(mean=mean, cov=cov, size=num_samples)
        if ax is None:
            _, ax = plt.subplots()
        for i in range(num_samples):
            ax.plot(x, y[i, :], color=color)

    def get_posterior(self, Xp, Xo, Yo):
        mup, kpp = self.compute_mu(Xp), self.compute_cov(Xp, Xp)
        muo, koo = self.compute_mu(Xo), self.compute_cov(Xo, Xo)
        kpo = self.compute_cov(Xp, Xo)
        K =  kpo.dot(np.linalg.inv(koo))
        mu = mup + K.dot(Yo-muo)
        cov = kpp - K.dot(kpo.T)
        return mu, cov
