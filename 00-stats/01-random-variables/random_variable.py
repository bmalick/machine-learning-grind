# implement random variables
# implement empirical statistics
# implement histograms

import numpy as np
import matplotlib.pyplot as plt

# distributions
def uniform_distribution(): return lambda n: np.random.rand(n)

def gaussian_distribution(mu: float = 0., sigma: float = 1.): return lambda n: np.random.normal(mu, sigma, n)

class RandomVariable:
    def __init__(self, name: str, distribution=uniform_distribution(), max_values=None):
        self.name = name
        self.distribution = distribution
        self.max_values = max_values

    @staticmethod
    def constant_var(c: float, name: str, max_values: int=None):
        return RandomVariable(name=name, distribution=lambda n: np.array([c]*n), max_values=max_values)

    def __call__(self, n: int = 5):
        if self.max_values is not None: n = self.max_values
        return self.distribution(n)

    def __str__(self):
        return f"{self.name} :\n" + "\n".join([f"   {v}" for v in self(4)])

    def __rsub__(self, r): return RandomVariable(name=f"{r}-{self.name}", distribution=lambda n: r-self(n))

    def __radd__(self, r): return RandomVariable(name=f"{r}+{self.name}", distribution=lambda n: r+self(n))

    def __rmul__(self, r): return RandomVariable(name=f"{r}*{self.name}", distribution=lambda n: r*self(n))

    def __rtruediv__(self, r): return RandomVariable(name=f"{self.name}/{r}", distribution=lambda n: self(n)/r)

    def __add__(self, other):
        if not isinstance(other, RandomVariable):
            other = RandomVariable.constant_var(other, f"{other}", self.max_values)
        return RandomVariable(name=f"{self.name}+{other.name}", distribution=lambda n: self(n)+other(n))

    def __sub__(self, other):
        if not isinstance(other, RandomVariable):
            other = RandomVariable.constant_var(other, f"{other}", self.max_values)
        return RandomVariable(name=f"{self.name}-{other.name}", distribution=lambda n: self(n)-other(n))

    def __mul__(self, other):
        if not isinstance(other, RandomVariable):
            other = RandomVariable.constant_var(other, f"{other}", self.max_values)
        return RandomVariable(name=f"{self.name}*{other.name}", distribution=lambda n: self(n)*other(n))

    def __truediv__(self, other):
        if not isinstance(other, RandomVariable):
            other = RandomVariable.constant_var(other, f"{other}", self.max_values)
        return RandomVariable(name=f"{self.name}/{other.name}", distribution=lambda n: self(n)/other(n))

    def __pow__(self, power: int): return RandomVariable(name=f"{self.name}^{power}", distribution=lambda n: self(n)**power)

    def expantacy(self, n: int = 1000): return np.mean(self(n), axis=0)

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

    def estimated_density(self, x: np.ndarray, kernel, n: int = 1000, method="scott"):
        sigma = self.std(n)
        if method=="scott": h = sigma**2 / n**(1/5) # KDE bandwidth
        elif method=="silverman": h = 0.9 * sigma**2 * n**(-0.2)

        samples = self(n)
        margin = (samples.max()-samples.min())*0.1
        if x is None:
            x = np.linspace(samples.min()-margin, samples.max()+margin, n)
        density = np.zeros(len(x))
        for xi in samples:
            density += kernel((x-xi) / h)
        return density / (n*h)

    @staticmethod
    def get_bivariate_density(var_x, var_y, n: int, ax=None, plot=True, method="scott", margin_factor=0.1, colors=None):
        gaussian_kernel = lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
        sigma_x = var_x.std(n)
        sigma_y = var_y.std(n)
        # Set KDE bandwidths based on method
        if method=="scott":
            hx = sigma_x * n**(-1/5) # Scott
            hy = sigma_y * n**(-1/5)
        elif method=="silverman":
            hx = 0.9 * sigma_x * n**(-0.2) # Silverman
            hy = 0.9 * sigma_y * n**(-0.2)

        sample_x = var_x(n)
        sample_y = var_y(n)
        margin_x = (sample_x.max() - sample_x.min()) * margin_factor
        margin_y = (sample_y.max() - sample_y.min()) * margin_factor
        x = np.linspace(sample_x.min() - margin_x, sample_x.max() + margin_x, n)
        y = np.linspace(sample_y.min() - margin_y, sample_y.max() + margin_y, n)
        xx, yy = np.meshgrid(x, y)

        density = np.zeros(xx.shape)
        for xi, yi in zip(sample_x, sample_y):
            kernel_x = gaussian_kernel((xx - xi) / hx) / hx
            kernel_y = gaussian_kernel((yy - yi) / hy) / hy
            density += kernel_x*kernel_y
        
        density /= n

        if plot:
            if ax is None: fig, ax = plt.subplots()
            # Plot the contour of the joint density
            if colors is not None:
                contour = ax.contour(xx, yy, density, levels=10, colors=colors)
            else:
                contour = ax.contour(xx, yy, density, levels=10)
            ax.clabel(contour, inline=True, fontsize=10)
            ax.set_title('p(%s, %s)' % (var_x.name, var_y.name))
            ax.set_xlabel(var_x.name)
            ax.set_ylabel(var_y.name)
                
        return xx, yy, density

