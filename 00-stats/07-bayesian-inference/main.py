#!/home/malick/miniconda3/envs/pt/bin/python3

import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x = np.linspace(0,1,100)

def get_pdf(x, alpha, beta):
    return scipy.stats.beta.pdf(x, alpha, beta)

def plot_pdfs(n0, n1, alpha, beta, ax):
    prior = get_pdf(x, alpha, beta)
    likelihood = get_pdf(x, n1+1, n0+1)
    posterior = scipy.stats.beta.pdf(x, alpha+n1, beta+n0)
    ax.plot(x, prior, "black", label=f"prior Beta({alpha},{beta})")
    ax.plot(x, likelihood, "r", linestyle="dotted", label=f"likelihood Beta({n1+1},{n0+1})")
    ax.plot(x, posterior, "b--", label=f"posterior Beta({alpha+n1},{beta+n0})")
    ax.legend()
    ax.set_title(f"n0={n0}, n1={n1}, alpha={alpha}, beta={beta}")

def test1():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_pdfs(n0=1, n1=4, alpha=1, beta=1, ax=axes[0])
    plot_pdfs(n0=1, n1=4, alpha=2, beta=2, ax=axes[1])
    plt.show()

def test2():

    fig, ax = plt.subplots()
    # Initial parameters
    alpha, beta = 2, 2  # Prior belief
    n0, n1 = 0, 0  # Initial number of tails and heads
    num_tosses = 50 # Number of tosses to simulate

    # Function to update the plot
    def update(frame):
        nonlocal n0, n1  # Keep track of heads and tails

        # Simulate a coin toss (1 = heads, 0 = tails)
        toss = np.random.choice([0, 1])
        if toss == 1:
            n1 += 1
        else:
            n0 += 1

        ax.clear()  # Clear previous plot

        # Compute PDFs
        prior = get_pdf(x, alpha, beta)
        likelihood = get_pdf(x, n1+1, n0+1)
        posterior = scipy.stats.beta.pdf(x, alpha+n1, beta+n0)

        # Plot the prior and posterior
        ax.plot(x, prior, "black", label=f"Prior Beta({alpha},{beta})")
        ax.plot(x, likelihood, "r", linestyle="dotted", label=f"likelihood Beta({n1+1},{n0+1})")
        ax.plot(x, posterior, "b-", label=f"Posterior Beta({alpha+n1},{beta+n0})")
        ax.legend()
        ax.set_title(f"Iteration {frame+1}: Toss={toss}, Heads={n1}, Tails={n0}")

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=num_tosses, repeat=False, interval=500)

    plt.show()


if __name__ == "__main__":
    functions = [test1, test2]
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
