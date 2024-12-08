import sys
import numpy as np
import matplotlib.pyplot as plt


#####################
# Method of Moments #
#####################
def moments_method(samples: np.array, k: int):
    def moment(i):
        return (samples**i).sum() / len(samples)
    return [moment(i) for i in range(1, k+1)]


# Application to gaussian variable
def estimate_gaussian_var_params(samples):
    moments = moments_method(samples, k=2)
    estimated_mu = moments[0]
    estimated_sigma = np.sqrt(moments[1]-moments[0])
    return estimated_mu, estimated_sigma

def gaussian_var_estimation():
    mu = 0.2
    sigma = 1.7
    print("Choosed params:")
    print("mu = %f - sigma = %f" % (mu, sigma))
    print("Estimation:")
    for n in [100, 1000, 10000]:
        gaussian_var = np.random.normal(loc=mu, scale=sigma, size=n)
        estimated_mu, estimated_sigma = estimate_gaussian_var_params(gaussian_var)
        print("n = %5d - mu = %.5f - sigma = %.5f" % (n, estimated_mu, estimated_sigma))

#####################
# Estimation of pi  #
#####################

def monte_carlo_pi():
    def compute_pi(n):
        pi = 0
        for _ in range(n):
            X = np.random.uniform(-1, 1, size=2)
            if X[0]**2 + X[1]**2 <=1: pi+=1
        return 4*pi/n
    for n in [10, 100, 1000, 10000]:
        print(n, compute_pi(n))

    fig, axes = plt.subplots(2,2, figsize=(19.2, 10.8))
    axes = axes.ravel()
    target = lambda x: (x[0]**2+x[1]**2 <= 1).astype(int)
    for i in range(1,5):
        n = 10**i
        x = np.random.uniform(-1,1,size=(n, 2))
        y = [target(t) for t in x]
        axes[i-1].scatter(x[:, 0], x[:, 1], s=10, c=y)
        axes[i-1].set_title(r"$n=%d - \hat{\pi}=%f$" % (n, compute_pi(n)))
    plt.show()





if __name__ == "__main__":
    functions = [
        gaussian_var_estimation,
        monte_carlo_pi,
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
