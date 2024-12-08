import sys
import numpy as np
import matplotlib.pyplot as plt

#######################################
# Pseudo-Random Number Generator      #
# Linear Congruential Generator (LCG) #
#######################################

def lcg(seed, a, c, m, n):
    numbers = []
    x = seed
    for _ in range(n):
        x = (a*x + c) % m
        numbers.append(x/m)
    return numbers

def test_lcg():
    for seed in [42, 43]:
        numbers = lcg(seed=seed, a=16464645, c=55466, m=2**32, n=10)
        print(f"Seed: {seed}, Numbers:\n", numbers)

#####################################
# Box-Muller Method                 #
# Sample from Gaussian distribution #
#####################################
def standard_normal_distribution(n: int):
    def box_muller():
        r = np.random.random()
        theta = np.random.random()
        return np.sqrt(-2 * np.log(r)) * np.cos(2*np.pi * theta)
    return np.array([box_muller() for _ in range(n)])

def normal_distribution(mean: float, std: float, n: int):
    y = standard_normal_distribution(n)
    return mean + std * y

def multuivariate_normal_distribution(mean: np.ndarray, sigma: np.ndarray, n: int):
    assert sigma.shape[0] == sigma.shape[1] == mean.shape[0]
    L = np.linalg.cholesky(sigma)
    def draw():
        y = standard_normal_distribution(len(mean)).reshape(-1,1)
        return mean.reshape(-1,1) + L@y
    return np.hstack([draw() for _ in range(n)])



def view_distribution(samples, ax):
    weights = np.ones_like(samples) / len(samples)
    ax.hist(samples, bins=30, color="gray", linewidth=0.5, alpha=0.7, density=True, weights=weights)

def compute_mean(samples):
    return samples.sum(axis=-1) / len(samples)

def compute_std(samples):
    return np.sqrt((samples**2).sum(axis=-1) / len(samples) - compute_mean(samples)**2)

def plot_estimate_distribution(samples, ax, n: int, kernel):
    sigma = compute_std(samples)
    # h = sigma**2 / n**(1/5)
    h = 0.9 * sigma**2 * n**(-0.2)

    margin = (samples.max() - samples.min()) * 0.1
    x = np.linspace(samples.min()-margin, samples.max()+margin, n)
    density = np.zeros(len(x))
    for xi in samples:
        density += kernel((x-xi) / h)
    density /= n*h
    ax.plot(x, density)
    return density


def test_std_normal():
    n = 1000
    x = standard_normal_distribution(n)
    mean = compute_mean(x)
    sigma = compute_std(x)
    print("mean: %f, sigma: %f" % (mean, sigma))

    fig, ax = plt.subplots()
    view_distribution(x, ax)
    kernel = lambda t: np.exp(-t**2 / 2) / np.sqrt(2*np.pi)
    plot_estimate_distribution(samples=x, ax=ax, n=1000, kernel=kernel)
    ax.set_title(r"Sampling $\mathcal{N}(0,1)$ with Box-Muller Method")
    plt.show()


def test_normal():
    n = 1000
    mean = 1.2
    sigma = 1.7
    x = normal_distribution(mean=mean, std=sigma, n=n)
    print("mean: %f, sigma: %f" % (compute_mean(x), compute_std(x)))

    fig, ax = plt.subplots()
    view_distribution(x, ax)
    kernel = lambda t: np.exp(-t**2 / 2) / np.sqrt(2*np.pi)
    plot_estimate_distribution(samples=x, ax=ax, n=1000, kernel=kernel)
    ax.set_title(r"Sampling $\mathcal{N}(%.2f,%.2f)$ with Box-Muller Method" % (mean, sigma))
    plt.show()

def test_mvn():
    n = 1000
    coef = 1.2
    # bias = -0.7
    mean_x = 2
    var_x = 4
    mean = np.array([mean_x, mean_x*coef])
    sigma = np.array([[var_x, coef*var_x], [coef*var_x, coef**2*var_x]])

    x = multuivariate_normal_distribution(mean=mean, sigma=sigma, n=n)
    noise = 0.1 * multuivariate_normal_distribution(mean=np.array([0, 0]), sigma=np.eye(2), n=n)
    x += noise

    fig, ax = plt.subplots()
    ax.scatter(x[0, :], x[1, :], alpha=0.5, s=10)
    ax.set_title(r"Multivariate normal sample from $(X,Y)$ with linear dependance")
    plt.show()


if __name__ == "__main__":
    functions = [
        test_lcg,
        test_std_normal, test_normal, test_mvn,
    ]
    if len(sys.argv) !=2:
        print("Usage: %s <function id>" % sys.argv[0])
        print()
        print("id | function")
        print("---+"+'-'*20)
        for id, f in enumerate(functions):
            print("%d  | %s" %(id, f.__name__))
        sys.exit()

    id = int(sys.argv[1])
    if(id < 0 or id >= len(functions)) :
        print("Function id %d is invalid (should be in [0, %d])" % (id, len(functions)-1))
        sys.exit()
    functions[id]()
