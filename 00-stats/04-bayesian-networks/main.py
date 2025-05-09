#!/home/malick/miniconda3/envs/pt/bin/python3

import numpy as np
import pandas as pd
import naive_bayes


if __name__ == "__main__":
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
    print("Categorical naive Bayes:")
    model = naive_bayes.CategoricalNaiveBayes()
    model.fit(data.iloc[:, :-1].values, data.y.values)
    x = np.array([np.array(xi) for xi in x_test.x.values])
    x_test["pred"] = x_test.x.map(lambda x: model.predict_proba(np.array(x)))
    print(x_test); print()
