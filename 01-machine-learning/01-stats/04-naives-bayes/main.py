#!/home/malick/miniconda3/envs/pt/bin/python3

import numpy as np
import pandas as pd
import naive_bayes


if __name__ == "__main__":
    x = np.random.normal(loc=1, scale=2, size=1000)
    mu, sigma = naive_bayes.mle_gaussian(x)
    print("MLE for Gaussian distribution:")
    print(f"mu={mu}, sigma={sigma}")
    print()

    x = np.random.binomial(n=1, p=0.3, size=1000)
    p = naive_bayes.mle_bernoulli(x)
    print("MLE for Bernoulli distribution:")
    print(f"p={p}")
    print()

    data = pd.DataFrame(
        data = {
            "earn": [1,0,0,1,0,1,1,0,1,0],
            "million": [1,0,1,1,0,0,0,0,0,1],
            "account": [0,1,1,0,0,0,0,0,1,1],
            "password": [0,1,0,0,0,0,0,1,1,1],
            "y": ["spam", "spam", "not spam", "spam", "not spam", "spam", "not spam", "spam", "spam", "not spam"],
        }
    )
    x_test = pd.DataFrame(
        data={
            "message": ["earn+million", "million+account", "account+password"],
            "x": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
        }
    )
    print("Naive Bayes:")
    model = naive_bayes.NaiveBayesFromScratch("bernoulli")
    model.fit(data.iloc[:, :-1].values, data.y.values)
    x = np.array([np.array(xi) for xi in x_test.x.values])
    x_test["pred"] = x_test.x.map(lambda x: model.predict_proba(np.array(x)))
    print(x_test); print()
