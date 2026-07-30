import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import animation

def gaussian_mixture(pis, mus, sigmas, N: int = 1000):
    assert len(pis)==len(mus)==len(sigmas), "K is not the same for pis, mus, sigmas"
    assert np.abs(1-pis.sum())<1.e-3, "probabilities weigths should sum to 1"
    K = pis.shape[0]
    ks = np.random.choice(K, p=pis, size=N)
    X =  np.vstack([np.random.multivariate_normal(mean=mus[i], cov=sigmas[i]) for i in ks])
    return X, ks

mus = np.array([[0.2, 4.3], [1.5, 2.3], [-2.8, -1.8], [-2.8, 2.8], [3.8, 1.1]])
sigmas = [
    np.diag([0.1,0.2]),
    np.diag([0.3,0.5]),
    np.diag([0.7,0.11]),
    np.diag([0.13,0.17]),
    np.diag([0.19,0.23]),
]

rotation_mat = lambda x: np.array([[np.cos(x), -np.sin(x)],[np.sin(x), np.cos(x)]])

rotations = [rotation_mat(x) for x in np.random.random(len(sigmas))]
rotated_sigmas = [R@S@R.T for (R,S) in zip(rotations, sigmas)]

pis = np.random.dirichlet(np.ones(mus.shape[0]))

print("sum of pis:", pis.sum())

N = 500
data, labels = gaussian_mixture(pis,mus,rotated_sigmas,N=N)
plt.scatter(data[:,0], data[:,1], c=labels, alpha=0.6)
plt.savefig("gaussian-mixture-data.jpg")
# plt.show()
plt.close()

sns.kdeplot(data)
plt.savefig("gaussian-mixture-kdeplot.jpg")
# plt.show()
plt.close()

def em_algorithm(K: int, X: np.ndarray, save_dir: str, eps_log: float = 1e-6, eps_conv: float = 1e-4, interval: int = 200):
    N = X.shape[0]
    ndim = X.shape[1]

    sigma_square = X.var()

    ELBO_old = - np.inf

    # Sample random values mu_1,mu_K from x1,...,xN
    mus = np.array([X[i] for i in np.random.randint(0, len(X), K)])

    sigmas = sigma_square * np.identity(ndim)
    sigmas = sigmas[np.newaxis,:].repeat(K, axis=0) / K

    params = []

    pis = np.ones(K) / K

    R = np.zeros((N,K))

    num_iter = 0

    while True:

        ELBO_new = 0

        # E step
        for i in range(N):
            xi = X[i]

            raw_p = np.zeros(K)

            for k in range(K):
                sigma_k = sigmas[k]
                mu_k = mus[k]
                # r_ik = p(zi = k|xi , \theta_t)
                # we use log trick here
                p_xi_k = -0.5 * (xi-mu_k).T @ np.linalg.inv(sigma_k) @ (xi-mu_k)
                p_xi_k -= 0.5 * np.log(np.linalg.det(sigma_k)+eps_log)
                p_xi_k -= 0.5 * ndim * np.log(2*np.pi + eps_log)

                r_ik = p_xi_k + np.log(pis[k]+eps_log)
                R[i,k] = r_ik

                raw_p[k] = r_ik
                # ELBO_new  = \sum_i \sum_k R[i,k] [log pis[k] + log p(xi|zi=k,\theta) - log R[i,k]]

            # max trick
            max_log = np.max(R[i,:])
            # exp-sum trick
            R[i,:] = np.exp(R[i,:] - max_log)
            R[i,:] /= R[i,:].sum()
            
            # compute ELBO
            ELBO_new += np.sum(R[i,:] * (raw_p - np.log(R[i,:] + eps_log)))

        # M step
        pis = R.sum(axis=0) / N
        mus = np.matmul(R.T, X) / R.sum(axis=0)[:,np.newaxis]

        for k in range(K):
            r_ik = R[:,k]
            r_k = r_ik.sum()
            pis[k] = r_k / N

            mus[k] = (r_ik[:, np.newaxis] * X).sum(axis=0) / r_k

            sigma_k = np.zeros((ndim,ndim))
            for i in range(N):
                diff = X[i] - mus[k]
                sigma_k += r_ik[i] * np.outer(diff, diff)
            sigma_k /= r_k

            sigmas[k] = sigma_k

        if np.abs(ELBO_new - ELBO_old) < eps_conv:
            break

        params.append((pis,mus,sigmas))

        ELBO_old = ELBO_new
        num_iter += 1

    fig, ax = plt.subplots()
    os.makedirs(save_dir, exist_ok=True)

    def init():
      sns.kdeplot(X[:,0], color="black", linestyle="--", ax=ax)
      sns.kdeplot(X[:,1], color="green", linestyle="--", ax=ax)

    def update_pdf(frame):
        ax.clear()
        init()
        temp_pis, temp_mus, temp_sigmas = params[frame]
        new_data, labels = gaussian_mixture(temp_pis,temp_mus,temp_sigmas)
        sns.kdeplot(new_data[:,1], color="b", ax=ax)
        sns.kdeplot(new_data[:,0], color="r", ax=ax)
        ax.legend(["true-pdf-0", "true-pdf-1", "est-pdf-0", "est-pdf-1"])
        ax.set_title(f"Iterations: {frame}")
        fig.savefig(os.path.join(save_dir, f"{frame}.png"))

    anim = animation.FuncAnimation(fig, update_pdf, frames=len(params), interval=interval)
    plt.close()

    print("Number of iterations:", num_iter)
    print("ELBO:", ELBO_new)
    return (pis, mus, sigmas), R, ELBO_new, anim

def test(K, X, eps_conv, interval=200):
    estimated_theta, probs, _, anim = em_algorithm(K=K, X=X, eps_conv=eps_conv, interval=interval, save_dir=f"em-K{K}")
    clusters = probs.argmax(axis=1)
    anim.save(f"./docs/em-K{K}.gif",writer='pillow')
    plt.title(f"culsters: K={K}")
    plt.scatter(data[:,0], data[:,1], c=clusters, alpha=0.6)
    plt.savefig(f"./docs/em-K{K}.jpg")
    plt.close()

for K in range(2,6):
    print("="*50)
    print(f"K={K}")
    test(K=K, X=data, eps_conv=1e-3, interval=200)

# TODO: EM algorithm: stochastic version
