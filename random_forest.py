from typing import Optional, Sequence, Mapping
import numpy as np
import pandas as pd
import random
from collections import Counter


class Node(object):
    def __init__(self, node_size: int, node_class: str, depth: int, single_class:bool = False):
        # Every node is a leaf unless you set its 'children'
        self.is_leaf = True
        # Each 'decision node' has a name. It should be the feature name
        self.name = None
        # All children of a 'decision node'. Note that only decision nodes have children
        self.children = {}
        # Whether corresponding feature of this node is numerical or not. Only for decision nodes.
        self.is_numerical = None
        # Threshold value for numerical decision nodes. If the value of a specific data is greater than this threshold,
        # it falls under the 'ge' child. Other than that it goes under 'l'. Please check the implementation of
        # get_child_node for a better understanding.
        self.threshold = None
        # The class of a node. It determines the class of the data in this node. In this assignment it should be set as
        # the mode of the classes of data in this node.
        self.node_class = node_class
        # Number of data samples in this node
        self.size = node_size
        # Depth of a node
        self.depth = depth
        # Boolean variable indicating if all the data of this node belongs to only one class. This is condition that you
        # want to be aware of so you stop expanding the tree.
        self.single_class = single_class

    def set_children(self, children):
        self.is_leaf = False
        self.children = children

    def get_child_node(self, feature_value)-> 'Node':
        if not self.is_numerical:
            return self.children[feature_value]
        else:
            if feature_value >= self.threshold:
                return self.children['ge'] # ge stands for greater equal
            else:
                return self.children['l'] # l stands for less than


class RandomForest(object):
    def __init__(self, n_classifiers: int,
                 criterion: Optional['str'] = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = None,
                 max_features: Optional[int] = None):
        """
        :param n_classifiers:
            number of trees to generated in the forrest
        :param criterion:
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “entropy” for the information gain.
        :param max_depth:
            The maximum depth of the trees.
        :param min_samples_split:
            The minimum number of samples required to be at a leaf node
        :param max_features:
            The number of features to consider for each tree.
        """
        self.n_classifiers = n_classifiers
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.criterion_func = self.entropy if criterion == 'entropy' else self.gini


    def fit(self, X: pd.DataFrame, y_col: str)->float:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of training dataset
        """
        features = self.process_features(X, y_col)
        # Your code
        for _ in range(self.n_classifiers):
            tree = self.generate_tree(X, y_col, features)
            self.trees.append(tree)
        return self.evaluate(X, y_col)

    def predict(self, X: pd.DataFrame)->np.ndarray:
        """
        :param X: data
        :return: aggregated predictions of all trees on X. Use voting mechanism for aggregation.
        """
        predictions = []
        # Your code
        for _, row in X.iterrows():
            tree_preds = [self.traverse_tree(tree, row) for tree in self.trees]
            most_common = Counter(tree_preds).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

    def evaluate(self, X: pd.DataFrame, y_col: str)-> int:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of predictions on X
        """
        preds = self.predict(X)
        acc = sum(preds == X[y_col]) / len(preds)
        return acc

    def generate_tree(self, X: pd.DataFrame, y_col: str,   features: Sequence[Mapping])->Node:
        """
        Method to generate a decision tree. This method uses self.split_tree() method to split a node.
        :param X:
        :param y_col:
        :param features:
        :return: root of the tree
        """
        root = Node(X.shape[0], X[y_col].mode(), 0)
        # Your code
        self.split_node(root, X, y_col, features)
        return root

    def split_node(self, node: Node, X: pd.DataFrame, y_col:str, features: Sequence[Mapping]) -> None:
        """
        This is probably the most important function you will implement. This function takes a node, uses criterion to
        find the best feature to slit it, and splits it into child nodes. I recommend to use revursive programming to
        implement this function but you are of course free to take any programming approach you want to implement it.
        :param node:
        :param X:
        :param y_col:
        :param features:
        :return:
        """
        if node.depth == self.max_depth or node.size <= self.min_samples_split or node.single_class:
            return

        best_feature = None
        best_threshold = None
        best_gini = float('inf')

        for feature in features:
            if feature['name'] == y_col:
                continue
            values = X[feature['name']].unique()
            for value in values:
                left_split = X[X[feature['name']] <= value]
                right_split = X[X[feature['name']] > value]
                gini = (len(left_split) / len(X)) * self.gini(left_split, feature['name'], y_col) + \
                    (len(right_split) / len(X)) * self.gini(right_split, feature['name'], y_col)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature['name']
                    best_threshold = value

        if best_gini == float('inf'):
            return

        node.name = best_feature
        node.threshold = best_threshold

        left_data = X[X[best_feature] <= best_threshold]
        right_data = X[X[best_feature] > best_threshold]

        if len(left_data) == 0 or len(right_data) == 0:
            return

        left_child = Node(left_data.shape[0], left_data[y_col].mode(), node.depth + 1)
        right_child = Node(right_data.shape[0], right_data[y_col].mode(), node.depth + 1)

        node.set_children({'l': left_child, 'ge': right_child})
        node.is_leaf = False

        self.split_node(left_child, left_data, y_col, features)
        self.split_node(right_child, right_data, y_col, features)


    def gini(self, X: pd.DataFrame, feature: Mapping, y_col: str) -> float:
        """
        Returns gini index of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """
        total_samples = len(X)
        gini_index = 1.0

        if total_samples == 0:
            return gini_index

        classes = X[y_col].unique()
        for c in classes:
            proportion = (len(X[X[y_col] == c]) / total_samples)
            gini_index -= proportion ** 2

        return gini_index
        #pass

    def entropy(self, X: pd.DataFrame, feature: Mapping, y_col: str) ->float:
        """
        Returns entropy of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """
        total_samples = len(X)
        entropy_val = 0.0

        if total_samples == 0:
            return entropy_val

        classes = X[y_col].unique()
        for c in classes:
            proportion = (len(X[X[y_col] == c]) / total_samples)
            entropy_val -= proportion * np.log2(proportion)

        return entropy_val
        #pass


    def process_features(self, X: pd.DataFrame, y_col: str)->Sequence[Mapping]:
        """
        :param X: data
        :param y_col: name of the label column in X
        :return:
        """
        features = []
        for n,t in X.dtypes.items():
            if n == y_col:
                continue
            f = {'name': n, 'dtype': t}
            features.append(f)
        return features