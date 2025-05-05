# Heart of Bayesian inference

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