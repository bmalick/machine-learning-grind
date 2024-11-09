# implement random variables
# implement empirical statistics
# implement histograms

import numpy as np
import matplotlib.pyplot as plt

# distributions
def uniform_distribution(): return lambda n: np.random.rand(n)

def gaussian_distribution(mu: float = 0., sigma: float = 1.): return lambda n: np.random.normal(mu, sigma, n)

class RandomVariable:
    def __init__(self, name: str, distribution=uniform_distribution()):
        self.name = name
        self.distribution = distribution

    @staticmethod
    def constant_var(c: float, name: str):
        return RandomVariable(name=name, distribution=lambda n: np.array([c]*n))

    def __call__(self, n: int = 5): return self.distribution(n)

    def __str__(self):
        return f"{self.name} :\n" + "\n".join([f"   {v}" for v in self(4)])

    def expantacy(self, n: int = 1000): return np.mean(self(n), axis=0)

    def __rsub__(self, r): return RandomVariable(name=f"{r}-{self.name}", distribution=lambda n: r-self(n))

    def __radd__(self, r): return RandomVariable(name=f"{r}+{self.name}", distribution=lambda n: r+self(n))

    def __add__(self, other):
        if not isinstance(other, RandomVariable):
            other = RandomVariable.constant_var(other, f"{other}")
        return RandomVariable(name=f"{self.name}+{other.name}", distribution=lambda n: self(n)+other(n))

    def __sub__(self, other):
        if not isinstance(other, RandomVariable):
            other = RandomVariable.constant_var(other, f"{other}")
        return RandomVariable(name=f"{self.name}-{other.name}", distribution=lambda n: self(n)-other(n))

    def __pow__(self, power: int): return RandomVariable(name=f"{self.name}^{power}", distribution=lambda n: self(n)**power)

    def variance(self, n: int = 1000): return ((self-self.expantacy(n))**2).expantacy(n)

    def std(self, n: int = 1000): return np.sqrt(self.variance(n))
    
    def apply(self, f): return RandomVariable(name=f"{self.name}_transformed", distribution = lambda n: np.array([f(x) for x in self(n)]))

    def apply_array(self, f): return RandomVariable(name=f"{self.name}_transformed", distribution = lambda n: f(self(n)))

    def nuplet(self, n: int = 5): return RandomVariable(name=f"{self.name}_{n}", distribution=lambda m: np.array([self(n) for _ in range(m)]))

    def plot(self, ax, n: int = 100, bins=30):
        samples = self(n)
        # if samples.shape[0]
        if len(samples.shape)!=1:
            print("Not available")
            return
        weights = np.ones_like(samples) / n
        ax.hist(samples, bins=bins, color="gray", edgecolor="black", linewidth=0.5, alpha=0.7, weights=weights, density=True)
        ax.set_xlabel(self.name)
        ax.set_ylabel("Density")

    def estimated_density(self, x: np.ndarray, kernel, n: int = 1000):
        h = 0.9 * self.variance(n) / n**0.2
        samples = self(n)
        x_grid = (x[:, np.newaxis] - samples) / h
        return kernel(x_grid).sum(axis=1) / (n * h)

