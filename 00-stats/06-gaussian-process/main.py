import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def zero_mean(x): return 0.

# Kernels
def squared_exp(sigma: float, l: float):
    return lambda x, y: sigma**2 * np.exp(-(x-y)**2 / (2*l**2))

def brownian():
    return lambda x,y: min(x,y)

class GaussianProcess:
    def __init__(self, mu, k):
        self.mu_name = mu.__qualname__
        self.k_name = k.__qualname__.split('.')[0]
        self.mu = np.vectorize(mu)
        self.k = np.vectorize(k)

    def sample(self, x: float, n: int=100):
        return np.random.normal(loc=self.mu(x), scale=self.k(x,x), size=n)

    def __call__(self, x):
        return self.sample(x, n=1)[0]

    def compute_mu(self, x): return self.mu(x)

    def compute_cov(self, x1, x2):
        return self.k(x1.reshape(-1,1), x2.reshape(1,-1))

    def represent_prior(self, num_samples: int=5, xmin: float=-5., xmax: float=5., n: int=100, ax=None):
        cmap = mpl.colormaps["Greys"]
        colors = cmap(np.linspace(0.3, 0.9, num_samples))
        x = np.linspace(xmin,xmax,n)
        cov = self.compute_cov(x,x)
        mean = self.compute_mu(x)
        y = np.random.multivariate_normal(mean=mean, cov=cov, size=num_samples)
        if ax is None:
            _, ax = plt.subplots()
        for i in range(num_samples):
            ax.plot(x, y[i, :], color=colors[i])
        ax.set_title(f"mean func: {self.mu_name} -- cov func: {self.k_name}")

    def get_posterior(self, Xp, Xo, Yo):
        mup, kpp = self.compute_mu(Xp), self.compute_cov(Xp, Xp)
        muo, koo = self.compute_mu(Xo), self.compute_cov(Xo, Xo)
        kpo = self.compute_cov(Xp, Xo)
        K =  kpo.dot(np.linalg.inv(koo))
        mu = mup + K.dot(Yo-muo)
        cov = kpp - K.dot(kpo.T)
        return mu, cov

def multivariate_normal(n: int = 1000):
    mean=np.array([0,0])
    cov = np.array([[1, 0.5], [0.5, 1.75]])
    X = np.random.multivariate_normal(mean=mean, cov=cov, size=n)

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    mins -= 0.1
    maxs += 0.1

    x = np.linspace(mins[0], maxs[0], 100)
    y = np.linspace(mins[1], maxs[1], 100)
    xx, yy = np.meshgrid(x, y)

    mean_hat = X.mean(axis=0)
    cov_hat = np.cov(X.T)

    dim = X.shape[1]
    def kernel(x):
        func = lambda x: np.exp(-0.5 * (x-mean_hat).T@np.linalg.inv(cov_hat)@(x-mean_hat)) / (np.sqrt((2*np.pi)**dim * np.linalg.det(cov_hat)))
        return np.array([func(t) for t in x])


    # # kernel = scipy.stats.gaussian_kde(X.T, bw_method="scott")
    xy = np.vstack([xx.ravel(), yy.ravel()])
    densities = kernel(xy.T).reshape(xx.shape)
    # plt.scatter(X[:, 0], X[:, 1])
    plt.contour(xx, yy, densities, levels=5)
    plt.savefig("./docs/multivariate-normal.jpg")
    # plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, densities, cmap="viridis", edgecolor="none")
    plt.savefig("./docs/multivariate-normal-3d.jpg")
    # plt.show()
    plt.close()

def test_gaussian_process():
    fig, axes = plt.subplots(1,2, figsize=(14,6))
    axes = axes.ravel()
    cov_functions = [squared_exp(sigma=1, l=2), brownian()]
    for i, func in enumerate(cov_functions):
        gaussian_process = GaussianProcess(
            mu=zero_mean, k=func
        )
        gaussian_process.represent_prior(ax=axes[i])
    plt.savefig("./docs/samples.jpg")
    # plt.show()
    plt.close()

target = lambda x : x ** 2 * np.exp(- np.abs(x)/3.)
f = np.vectorize(target)

def gaussian_process_regression(axes, num_observations=5, num_samples=3, noise_data=0, kernel=squared_exp(sigma=1, l=1)):
    Xo = np.random.uniform(-5,5,num_observations)
    Yo = f(Xo) + np.random.normal(0., noise_data, Xo.shape[0])
    Xp = np.linspace(-5,5,100)

    gaussian_process = GaussianProcess(
        mu=lambda x: 0.,
        k=kernel,
    )
    gaussian_process.represent_prior(ax=axes[0], num_samples=num_samples)

    mu, cov = gaussian_process.get_posterior(Xp=Xp, Xo=Xo, Yo=Yo)
    y = np.random.multivariate_normal(mean=mu, cov=cov, size=num_samples)
    for i in range(num_samples):
        axes[1].plot(Xp, y[i,:])
    # axes[1].plot(Xo, Yo, 'x', markeredgecolor=color,  markeredgewidth=2, markersize=10)
    axes[1].plot(Xo, Yo, 'x', markeredgewidth=2, markersize=10)
    std = np.sqrt(np.diag(cov))  # Extract standard deviation
    axes[1].fill_between(Xp, mu - 2 * std, mu + 2 * std, color='gray', alpha=0.3, linewidth=0.1)  # Shaded region
    axes[1].plot(Xp, mu, 'k', lw=1)  # Plot mean
    axes[0].set_title(f"{num_samples} samples")
    axes[1].set_title(f"{num_observations} observations")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])


def gaussian_reg():
    fig, axes = plt.subplots(4,2,figsize=(10,8))
    for i, n in enumerate([5, 10, 50, 100]):
        gaussian_process_regression(axes=axes[i,:], num_observations=n, num_samples=3, noise_data=0., kernel=squared_exp(sigma=1, l=1))
    plt.savefig("./docs/regression.jpg")
    # plt.show()
    plt.close()


def gaussian_process_regression_white_noise(axes, num_observations=5, num_samples=3, noise_data=0, white_noise=0.1, kernel=squared_exp(sigma=1, l=1)):
    Xo = np.random.uniform(-5,5,num_observations)
    Yo = f(Xo) + np.random.normal(0., noise_data, Xo.shape[0])
    Xp = np.linspace(-5,5,100)

    white_noise_cov_func = lambda x,y: white_noise**2 if np.abs(x-y)<1e-8 else 0.
    gaussian_process = GaussianProcess(
        mu=lambda x: 0.,
        k=kernel)

    gaussian_process.represent_prior(ax=axes[0], num_samples=num_samples)
    mu, cov = gaussian_process.get_posterior(Xp=Xp, Xo=Xo, Yo=Yo)
    y = np.random.multivariate_normal(mean=mu, cov=cov, size=num_samples)
    for i in range(num_samples):
        axes[1].plot(Xp, y[i,:])
    axes[1].plot(Xo, Yo, 'x', markeredgewidth=2, markersize=10)
    std = np.sqrt(np.diag(cov))  # Extract standard deviation
    axes[1].fill_between(Xp, mu - 2 * std, mu + 2 * std, color='gray', alpha=0.3, linewidth=0.1)  # Shaded region
    axes[1].plot(Xp, mu, 'k', lw=1)  # Plot mean
    axes[0].set_title(f"{num_samples} samples")
    axes[1].set_title(f"{num_observations} observations")
    
    gaussian_process_with_noise = GaussianProcess(
        mu=lambda x: 0.,
        k=lambda x,y: kernel(x,y) + white_noise_cov_func(x,y)
    )

    gaussian_process_with_noise.represent_prior(ax=axes[2], num_samples=num_samples)
    mu, cov = gaussian_process_with_noise.get_posterior(Xp=Xp, Xo=Xo, Yo=Yo)
    y = np.random.multivariate_normal(mean=mu, cov=cov, size=num_samples)
    for i in range(num_samples):
        axes[3].plot(Xp, y[i,:])
    axes[3].plot(Xo, Yo, 'x', markeredgewidth=2, markersize=10)
    std = np.sqrt(np.diag(cov))  # Extract standard deviation
    axes[3].fill_between(Xp, mu - 2 * std, mu + 2 * std, color='gray', alpha=0.3, linewidth=0.1)  # Shaded region
    axes[3].plot(Xp, mu, 'k', lw=1)  # Plot mean
    axes[2].set_title(f"with noise - {num_samples} samples")
    axes[3].set_title(f"with noise - {num_observations} observations")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

def gaussian_reg_noise():
    fig, axes = plt.subplots(4,4,figsize=(15,10.8))
    for i, n in enumerate([5, 10, 50, 100]):
        gaussian_process_regression_white_noise(axes=axes[i,:], num_observations=n, num_samples=3, noise_data=0., white_noise=0.1, kernel=squared_exp(sigma=1, l=1))
    # plt.show()
    plt.savefig("./docs/regression-noise.jpg")
    plt.close()


multivariate_normal()
test_gaussian_process()
gaussian_reg()
gaussian_reg_noise()
