import sys
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import decision_tree
import utils

def train_classification(model):
    model.fit(X=iris.data[:, 2:], y=iris.target)
    acc = utils.accuracy(y_pred=model.predict(iris.data[:, 2:]), y_true=iris.target)
    risk = utils.empirical_risk(model, iris.data[:, 2:], iris.target)
    print(model)
    print("Empirical risk:", risk)
    print("Train accuracy:", acc)
    print()
    model.show()

def train_regression(model):
    X, y = utils.create_dataset(n=500, noise=0.5)
    model.fit(X, y)
    X_test, y_test = utils.create_dataset(n=100)
    risk_func = lambda y1,y2: ((y1-y2)**2).mean() 
    print("Empirical risk:", risk_func(model.predict(X),y))
    print("Test risk:", risk_func(model.predict(X_test),y_test))
    model.show()
    plt.scatter(X, y, s=10, alpha=0.5)
    plt.scatter(X, model.predict(X), s=2, alpha=0.5)
    plt.show()

if __name__ == "__main__":
    iris = load_iris()
    clf_models = [
        decision_tree.DecisionTree(cost_function=utils.gini),
        decision_tree.DecisionTree(cost_function=utils.gini, max_depth=10),
        decision_tree.DecisionTree(cost_function=utils.gini, max_depth=10, min_samples_per_leaf=50)
    ]
    # for m in clf_models: train_classification(m)

    X_train, y_train = utils.create_dataset(n=500, noise=0.5)
    reg_models = [
        decision_tree.DecisionTree(cost_function=utils.mse),
        decision_tree.DecisionTree(cost_function=utils.mse, max_depth=10) # overfitting case
    ]
    # for m in reg_models: train_regression(m)

    models = clf_models + reg_models

    def train(idx):
        if idx <3: train_classification(models[idx])
        else: train_regression(models[idx])

    if len(sys.argv) !=2:
        print("Usage: %s <function id>" % sys.argv[0])
        print()
        print("id | function")
        print("---+"+'-'*20)
        for id, m in enumerate(models):
            print("%d  | %s" %(id, str(m)))
        sys.exit()

    id = int(sys.argv[1])
    if(id < 0 or id >= len(models)) :
        print("Function id %d is invalid (should be in [0, %d])" % (id, len(models)-1))
        sys.exit()
    train(id)

