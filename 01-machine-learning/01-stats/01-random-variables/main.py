#!/home/malick/miniconda3/envs/pt/bin/python
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy import stats
from random_variable import RandomVariable, gaussian_distribution

def test_var(var, n=1000):
    print(var)
    print(f"Expantacy(n={n}) = ", var.expantacy(n))
    print(f"Variance(n={n}) = ", var.variance(n))
    print(f"std(n={n}) = ", var.std(n))
    print()


def some_random_var():
    print("Constante variable")
    const = RandomVariable.constant_var(1,"1")
    test_var(const)

    print("Random variable with normal law")
    X = RandomVariable("X", gaussian_distribution(mu=-1.85, sigma=3.14))
    test_var(X)

    print("nuplet of the random variable above")
    X_n = X.nuplet(3)
    test_var(X_n)

    X_minus_X = X - X
    test_var(X_minus_X)

    X_plus_X = X + X
    test_var(X_plus_X)

    X_pow_2 = X**2
    test_var(X_pow_2)


def test_plot():
    print("Random variable with normal law")
    mu    = 0
    sigma = 1

    var = RandomVariable("X", gaussian_distribution(mu=mu, sigma=sigma))
    t = np.linspace(-4,4,500)
    real_density = np.exp(-(t-mu)**2 / 2 / sigma**2) / np.sqrt(2 * np.pi) / sigma
    gaussian_kernel = lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

    plt.figure(figsize=(19.20,10.80))
    plt.title("Estimation of density")
    for i in range(1, 4):
        N = 10**(i+1)
        ax = plt.subplot(2,2,i)
        estimated_density = var.estimated_density(x=t, kernel=gaussian_kernel, n=N)
        ax.plot(t, real_density, c="black", label="real density")
        ax.plot(t, estimated_density, c="black", linestyle="dotted", label="kernel-based estimated density")
        var.plot(ax=ax, n=N)
        ax.legend()
    plt.show()

def bivariate_distribution():
    n = 100
    var_x = RandomVariable("X", gaussian_distribution(mu=0, sigma=1))
    var_y = RandomVariable("Y", gaussian_distribution(mu=0, sigma=1))
    # var_y = 2*var_x + RandomVariable("Noise", gaussian_distribution(mu=0, sigma=1))
    fig, ax = plt.subplots()
    RandomVariable.get_bivariate_density(var_x=var_x, var_y=var_y, n=n, ax=ax)
    plt.show()

def bivariate_distribution_configs():
    n = 100
    fig, axes = plt.subplots(2,2,figsize=(19.2,10.8))
    axes = axes.ravel()

    x = np.random.normal(0,1,1000)
    var_x = RandomVariable("X", lambda n: x[:n])
    y = 2*x+np.random.normal(0,1,1000)
    var_y = RandomVariable("2*X+noise", lambda n: y[:n])
    # var_y = 2*var_x + RandomVariable("Noise", gaussian_distribution(mu=0, sigma=1))
    RandomVariable.get_bivariate_density(var_x=var_x, var_y=var_y, n=n, ax=axes[0]) # Positive correlation
    axes[0].set_title("Positive correlation")
    
    y = -2*x+np.random.normal(0,1,1000)
    var_y = RandomVariable("-2*X+noise", lambda n: y[:n])
    RandomVariable.get_bivariate_density(var_x=var_x, var_y=var_y, n=n, ax=axes[1]) # Negative correlation
    axes[1].set_title("Negative correlation")

    var_x = RandomVariable("X", gaussian_distribution(mu=0, sigma=1))
    var_y = RandomVariable("X", gaussian_distribution(mu=0, sigma=1))
    RandomVariable.get_bivariate_density(var_x=var_x, var_y=var_y, n=n, ax=axes[2]) # No correlation
    axes[2].set_title("No correlation")

    x = np.random.normal(0,1,1000)
    var_x = RandomVariable("X", lambda n: x[:n])
    y = 2*x+np.random.normal(0,1,1000)
    var_y = RandomVariable("2*X+noise", lambda n: y[:n])
    RandomVariable.get_bivariate_density(var_x=var_x, var_y=var_y, n=n, ax=axes[3])

    x = np.random.normal(0,0.5,1000)
    var_x = RandomVariable("X", lambda n: x[:n])
    y = 2*x+np.random.normal(0,0.5,1000)
    var_y = RandomVariable("2*X+noise", lambda n: y[:n])
    RandomVariable.get_bivariate_density(var_x=var_x, var_y=var_y, n=n, ax=axes[3], colors="#ff7f0e")
    x = np.random.normal(0,0.001,1000)
    var_x = RandomVariable("X", lambda n: x[:n])
    y = 2*x+np.random.normal(0,0.001,1000)
    var_y = RandomVariable("2*X+noise", lambda n: y[:n])
    RandomVariable.get_bivariate_density(var_x=var_x, var_y=var_y, n=n, ax=axes[3])
    axes[3].set_title("Low and high variance")
    axes[3].set_xlabel("")
    axes[3].set_ylabel("")
    
    plt.show()

def diabete_freatures():
    # Get dataset from https://www.kaggle.com/datasets/mathchi/diabetes-data-set?resource=download
    df_init = pd.read_csv('diabetes.csv')
    df_init.rename(columns={'BloodPressure': 'Pressure', 'SkinThickness': 'Skin', 'DiabetesPedigreeFunction': 'Pedigree', 'Outcome':'Diabetes'}, inplace=True)
    features_to_cleanse = ['Glucose','Pressure','Skin','Insulin','BMI']
    df_init[features_to_cleanse] = df_init[features_to_cleanse].replace(0, np.nan)
    gaussian_kernel = lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    # n = len(df)
    n = 100
    columns = df_init.columns[:-1]
    fig, axes = plt.subplots(8,8, figsize=(19.2, 10.8))

    for i in range(len(columns)):
        for j in range(len(columns)):
            var1 = columns[i]
            var2 = columns[j]
            if i!=j:
                df = df_init.loc[:, [var1,var2]].dropna()
            else: df = df_init.loc[:, [var1]].dropna()
            var_x = RandomVariable(var1, distribution=lambda n: df[var1][:n].values, max_values=len(df[var1]))
            var_y = RandomVariable(var2, distribution=lambda n: df[var2][:n].values, max_values=len(df[var2]))
            plot = True
            ax = axes[i, j]
            if i==j:
                plot = False
            xx, yy, joint_density = RandomVariable.get_bivariate_density(var_x=var_x, var_y=var_y, n=n, ax=ax, plot=plot, method="scott")
            if not plot: ax.plot(var_x.estimated_density(None, gaussian_kernel, n))
            ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
    for i in range(len(columns)): axes[i,0].set_ylabel(columns[i])
    for i in range(len(columns)): axes[7,i].set_xlabel(columns[i])
    plt.show()

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
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, densities, cmap="viridis", edgecolor="none")
    plt.show()


if __name__ == "__main__":
    functions = [
        some_random_var, test_plot,
        bivariate_distribution, bivariate_distribution_configs,
        diabete_freatures, multivariate_normal
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
