import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from graphviz import Digraph

class Node:
    def __init__(self, feature=None, label=None, threshold=None, samples=None, cost_function=None):
        self.label = label
        self.feature = feature
        self.threshold = threshold
        self.samples = samples if samples is not None else []
        self._parent = None
        self._left = None
        self._right = None
        self.cost_function = cost_function

    def __len__(self): return len(self.samples[0])

    @property
    def parent(self): return self._parent

    @parent.setter
    def parent(self, node): self._parent = node

    @property
    def left(self): return self._left

    @left.setter
    def left(self, node): self._left = node

    @property
    def right(self): return self._right

    @right.setter
    def right(self, node): self._right = node

    def __str__(self):
        value = {yi: 0 for yi in self.samples[1]}
        for yi in self.samples[1]: value[yi] += 1
        value = ", ".join(["%s: %d" % (k,v) for k,v in value.items()])

        res = "gini: %.5s\nfeature: %s\nthreshold: %.4s\nsamples: %d\nvalue: [%s]\nclass: %s" % (
            self.cost_function(self.samples), str(self.feature), str(self.threshold), len(self), value, str(self.label),
        )
        return res
    
    def __repr__(self): return self.__str__()


class DecisionTree:
    def __init__(self, cost_function,
                 max_depth: int = 2,
                 min_samples_per_leaf: int = None,
                 min_samples_per_split: int = None,
                 max_features: int = None,
                 pure_node: bool = False,
                 max_leaf_nodes: int = None):
        self.cost_function = cost_function
        self.max_depth = max_depth
        self.min_samples_per_leaf = min_samples_per_leaf
        self.min_samples_per_split = min_samples_per_split
        self.max_features = max_features
        self.pure_node = pure_node
        self.max_leaf_nodes = max_leaf_nodes
    
    def __str__(self):
        return "DecisionTree(cost=%s, max_depth=%s, min_samples_per_leaf=%s, min_samples_per_split=%s, "\
            "max_features=%s, pure_node=%s, max_leaf_nodes=%s)" % (
                self.cost_function.__name__, self.max_depth, self.min_samples_per_leaf,
                self.min_samples_per_split, self.max_features, self.pure_node, self.max_leaf_nodes
            )

    def fit(self, X, y):
        self.tree = DecisionTree.growing_tree(
            data=(X,y), cost_function=self.cost_function, depth=0,
            max_depth=self.max_depth,
            min_samples_per_leaf=self.min_samples_per_leaf
        )

    def find_node(self, x: np.ndarray) -> Node:
        if not hasattr(self, "tree"): return
        root = self.tree
        while root.left is not None and root.right is not None:
            if x[root.feature] <= root.threshold:
                root = root.left
            else: root = root.right
        return root

    def predict(self, x: np.ndarray) -> int|np.ndarray:
        if len(x.shape)==1:
            return self.find_node(x).label
        else: return np.array([self.predict(xi) for xi in x])
    
    def predict_probas(self, x: np.ndarray) -> float|np.ndarray:
        if len(x.shape)==1:
            node = self.find_node(x)
            # print(node.samples[0])
            return len(node.samples[1]==node.label) / len(node.samples[0])
        else: return np.array([self.predict_probas(xi) for xi in x])
    
    def show(self):
        if hasattr(self, "tree"): DecisionTree.plot_tree(self.tree)
        else: return

    @staticmethod
    def growing_tree(data, cost_function, depth: int, max_depth: int = None,
                     min_samples_per_leaf: int = None):
        _, y = data

        if len(set(y))==1 or len(y) == 0 or \
            (max_depth is not None and max_depth<=depth) or \
            (min_samples_per_leaf is not None and len(y) < min_samples_per_leaf):
            return Node(feature=None, label=DecisionTree.get_max_feature(y), threshold=None, samples=data, cost_function=cost_function)

        feature, threshold, left_node, right_node = DecisionTree.node_split(data, cost_function)

        if feature is None:
            return Node(feature=feature, threshold=threshold, samples=data, cost_function=cost_function)
        
        root = Node(feature=feature, threshold=threshold, samples=data, cost_function=cost_function)
        root.left = left_node
        root.right = right_node
        left_node.parent = root
        right_node.parent = root

        if min_samples_per_leaf is not None:
            if len(left_node.samples[0]) < min_samples_per_leaf:
                root.left = Node(feature=None, label=DecisionTree.get_max_feature(left_node.samples[1]), threshold=None, samples=left_node.samples, cost_function=cost_function)
            else:
                root.left = DecisionTree.growing_tree(left_node.samples, cost_function, depth + 1, max_depth, min_samples_per_leaf)
            if len(right_node.samples[0]) < min_samples_per_leaf:
                root.right = Node(feature=None, label=DecisionTree.get_max_feature(right_node.samples[1]), threshold=None, samples=right_node.samples, cost_function=cost_function)
            else:
                root.right = DecisionTree.growing_tree(right_node.samples, cost_function, depth + 1, max_depth, min_samples_per_leaf)
        else:
            root.left = DecisionTree.growing_tree(left_node.samples, cost_function, depth + 1, max_depth, min_samples_per_leaf)
            root.right = DecisionTree.growing_tree(right_node.samples, cost_function, depth + 1, max_depth, min_samples_per_leaf)

        root.left.parent = root
        root.right.parent = root

        return root
    
    @staticmethod
    def node_split(data, cost_function):
        X, y = data
        best_feature = None
        best_threshold = None
        left_node = None
        right_node = None
        best_cost = np.inf

        for f in range(X.shape[1]):
            thresholds = (np.unique(X[:, f])[:-1] + np.unique(X[:, f])[1:]) / 2

            for t in thresholds:
                left_indices = X[:, f] <= t
                right_indices = X[:, f] > t

                l_node = (X[left_indices], y[left_indices])
                r_node = (X[right_indices], y[right_indices])

                total_samples = len(y)
                left_weight = len(l_node[1]) / total_samples
                right_weight = len(r_node[1]) / total_samples
                split_cost = (left_weight * cost_function(l_node)) + (right_weight * cost_function(r_node))

                if split_cost < best_cost:
                    best_cost = split_cost
                    best_feature = f
                    best_threshold = t
                    left_node = l_node
                    right_node = r_node

        left_node = Node(feature=None, samples=left_node, cost_function=cost_function)
        right_node = Node(feature=None, samples=right_node, cost_function=cost_function)

        return best_feature, best_threshold, left_node, right_node
    
    @staticmethod
    def get_max_feature(y):
        return max(set(y), key=list(y).count)

    @staticmethod
    def plot_tree(root, filename="decision-tree"):
        dot = Digraph(comment='Decision Tree')

        def add_nodes_edges(node, dot, parent=None):
            if node is None:
                return

            node_label = str(node)
            dot.node(str(id(node)), label=node_label, shape='box', style='filled', color='lightblue')

            if parent:
                dot.edge(str(id(parent)), str(id(node)))

            if node.left is not None:
                add_nodes_edges(node.left, dot, node)
            if node.right is not None:
                add_nodes_edges(node.right, dot, node)

        add_nodes_edges(root, dot)

        dot.render(filename, format='png', cleanup=True)

        img = mpimg.imread(f"{filename}.png")
        plt.figure(figsize=(19.2, 10.8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

