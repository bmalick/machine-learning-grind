# import time

from sklearn.datasets import load_iris
import decision_tree
import utils

def train(model, show=False):
    model.fit(X=iris.data[:, 2:], y=iris.target)
    acc = utils.accuracy(y_pred=model.predict(iris.data[:, 2:]), y_true=iris.target)
    risk = utils.empirical_risk(model, iris.data[:, 2:], iris.target)
    print(model)
    print("Empirical risk:", risk)
    print("Train accuracy:", acc)
    print()
    if show: model.show()

if __name__ == "__main__":
    iris = load_iris()
    models = [decision_tree.DecisionTree(cost_function=utils.gini),
              decision_tree.DecisionTree(cost_function=utils.gini, max_depth=10),
              decision_tree.DecisionTree(cost_function=utils.gini, max_depth=10, min_samples_per_leaf=50)
            ]
    for m in models: train(m, True)
