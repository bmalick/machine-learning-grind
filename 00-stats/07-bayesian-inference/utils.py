import scipy
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

def get_pdf(x, alpha, beta):
    return scipy.stats.beta.pdf(x, alpha, beta)

def plot_pdfs(n0, n1, alpha, beta, ax):
    x = np.linspace(0,1,100)
    prior = get_pdf(x, alpha, beta)
    likelihood = get_pdf(x, n1+1, n0+1)
    posterior = scipy.stats.beta.pdf(x, alpha+n1, beta+n0)
    ax.plot(x, prior, "black", label=f"prior Beta({alpha},{beta})")
    ax.plot(x, likelihood, "r", linestyle="dotted", label=f"likelihood Beta({n1+1},{n0+1})")
    ax.plot(x, posterior, "b--", label=f"posterior Beta({alpha+n1},{beta+n0})")
    ax.legend()
    ax.set_title(f"n0={n0}, n1={n1}, alpha={alpha}, beta={beta}")

def bayesian_inference(alpha, beta, num_tosses=100, jupyter_notebook=False):
    x = np.linspace(0,1,100)

    fig, ax = plt.subplots()
    # Initial parameters
    alpha, beta = alpha, beta  # Prior belief
    mean = alpha / (alpha + beta)
    n0, n1 = 0, 0  # Initial number of tails and heads
    num_tosses = num_tosses # Number of tosses to simulate

    prior = get_pdf(x, alpha, beta)

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
        likelihood = get_pdf(x, n1+1, n0+1)
        posterior = scipy.stats.beta.pdf(x, alpha+n1, beta+n0)

        # Plot the prior and posterior
        ax.plot(x, prior, "black", label=f"Prior Beta({alpha},{beta})")
        ax.plot(x, likelihood, "r", linestyle="dotted", label=f"likelihood Beta({n1+1},{n0+1})")
        ax.plot(x, posterior, "b-", label=f"Posterior Beta({alpha+n1},{beta+n0})")
        ax.legend()
        ax.set_title(f"Iteration {frame+1}: Toss={toss}, Heads={n1}, Tails={n0}")

        ymin, ymax = ax.get_ylim()
        ax.plot([mean, mean], [ymin, ymax], linestyle="--", color="black")

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=num_tosses, repeat=False, interval=100)

    if jupyter_notebook: HTML(ani.to_jshtml())
    else: plt.show()
