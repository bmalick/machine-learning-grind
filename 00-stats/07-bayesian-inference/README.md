# Bayesian inference

> [!Note]
> Here is my [PDF note](https://bmalick.github.io/machine-learning-grind/01-estimation-and-bayesian-inference.pdf)

### Heart of Bayesian inference

As we treat parameters $\theta$ as unknown and data $\mathcal{D}$ fixed and known, we represent our uncertainty about parameters, after (posterior to) seeing data, by computing the **posterior distribution** using Bayes' rule:

$$
p(\theta | \mathcal{D}) = \frac{p(\theta) p(\mathcal{D} | \theta)}{p(\mathcal{D})} = \frac{p(\theta) p(\mathcal{D} | \theta)}{\int_{\boldsymbol{\theta}'} p(\mathcal{D}|\boldsymbol{\theta}')p(\boldsymbol{\theta}')}
$$

where $p(\theta)$ is the **prior** and represents our beliefs about the parameters before seeing the data;  
$p(\mathcal{D} | \theta)$ is called the **likelihood**, and represents our beliefs about what data we expect to see for each setting of the parameters;  
$p(\theta | \mathcal{D})$ is called the **posterior**, and represents our beliefs about the parameters after seeing the data;  
and $p(\mathcal{D})$ is called the **marginal likelihood** or **evidence**, and is a normalization constant.  
The task of computing this posterior is called **Bayesian inference**, **posterior inference**, or just **inference**.

For likelihood, we assume the data are **i.i.d.** (independent and identically distributed).  
For the prior, we assume we know nothing about the parameter.  
We note $p(\theta) = p(\theta|\kappa)$, with $\kappa$ called **hyperparameters** since they are parameters of the prior which determine our beliefs about the **main** parameter $\theta$.

> **Remark**
> 
> - Likelihood is not a distribution. It measures the relative plausibility of parameters given data.  
> - i.i.d. does not mean samples are independent. In practice, hidden dependencies may exist due to underlying structures or data collection methods.

Bayesian inference offers several advantages, including the integration of prior knowledge and the ability to quantify uncertainty. For example, in a network server problem where processing times depend on query size, Bayesian methods can model uncertainties in both processing times and query sizes. However, practical challenges such as selecting priors and computational complexity must be addressed. Common solutions include using conjugate priors for analytical tractability or applying numerical methods like Markov Chain Monte Carlo (MCMC).

Bayesian inference with multiple samples has two properties:
- processing of observations can be done one by one and
- processing observations can be done in any order


$$
\begin{align}
p(\theta \mid x_1, x_2) 
&\propto p(\theta) L(x_1, x_2 ; \theta) \\
&\propto p(\theta) L(x_1; \theta) L(x_2; \theta) \\
&\propto L(x_2; \theta) \left(L(x_1; \theta) p(\theta)\right) = L(x_2; \theta) p(\theta \mid x_1) \\
&\propto L(x_1; \theta) \left(L(x_2; \theta) p(\theta)\right) = L(x_1; \theta) p(\theta \mid x_2)
\end{align}
$$

Bayesian inference is naturally **online**.  
The posterior contains all model information about past observations.

### The problem of choosing a prior

The choice of a prior distribution $p(\theta)$ is a fundamental aspect of Bayesian inference. Philosophically, the prior represents existing knowledge before observing the data. When little prior knowledge is available, a more dispersed or non-informative prior is used, whereas a well-chosen prior can significantly impact the speed and quality of convergence.

A crucial aspect of priors is their connection to regularization, particularly in scenarios with limited data. The necessity of a prior often leads to criticism of Bayesian methods for introducing subjectivity. However, priors also provide flexibility and structure in probabilistic modeling, distinguishing Bayesian approaches from frequentist methods. The discussion between Bayesian and frequentist perspectives will be explored further in the course.

In practical applications, prior selection involves a trade-off. One option is to choose a tractable but potentially unrealistic prior, which enables exact inference. This is the case with conjugate priors, which simplify computations by ensuring that the posterior remains within the same family of distributions as the prior. This property is particularly useful in Bayesian updating, as it allows for analytical solutions and computational efficiency. However it does not fit reality.

Another approach is to use a more representative but intractable prior, requiring approximate inference techniques such as Markov Chain Monte Carlo (MCMC) sampling or variational inference.
