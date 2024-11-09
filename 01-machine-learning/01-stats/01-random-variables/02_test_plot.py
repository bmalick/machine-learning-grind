import numpy as np
import matplotlib.pyplot as plt
from random_variable import RandomVariable, gaussian_distribution

if __name__ == "__main__":
    print("Random variable with normal law")
    mu    = 0
    sigma = 1

    var = RandomVariable("X", gaussian_distribution(mu=mu, sigma=sigma))
    t = np.linspace(-4,4,500)
    real_density = np.exp(-(t-mu)**2 / 2 / sigma**2) / np.sqrt(2 * np.pi) / sigma
    gaussian_kernel = lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

    plt.figure(figsize=(19.20,10.80))
    for i in range(1, 4):
        N = 10**(i+1)
        ax = plt.subplot(2,2,i)
        estimated_density = var.estimated_density(x=t, kernel=gaussian_kernel, n=N)
        ax.plot(t, real_density, c="black", label="real density")
        ax.plot(t, estimated_density, c="black", linestyle="dotted", label="kernel-based estimated density")
        var.plot(ax=ax, n=N)
        ax.legend()
    plt.show()
