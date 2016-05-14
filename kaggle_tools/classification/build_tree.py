# Build a decsion tree in the correct format to be visualized
#  This script is mostly inspired from this blog post:
# http://planspace.org/20151129-see_sklearn_trees_with_d3/

import json
import os

from sklearn import tree
from sklearn.datasets import load_iris


def build_tree_rules(clf, features, labels, node_index=0):
    """Structure of rules in a fit decision tree classifier

    Args
    clf (sklearn model): DecisionTreeClassifier
        A tree that has already been fit.

    features (List[str]), labels(List[str]):
        The names of the features and labels, respectively.

    Retruns:
        A JSON tree structure
    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} > {}'.format(feature, threshold)
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        # Recursively build the children nodes
        node['children'] = [build_tree_rules(clf, features, labels, right_index),
                            build_tree_rules(clf, features, labels, left_index)]
    return node

# An example using the iris dataset

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)


tree_rules = build_tree_rules(clf, iris.feature_names,
                              iris.target_names)

file_name = "tree_structure.json"
file_path = os.path.join(os.path.abspath("classification"), file_name)
with open(file_path, 'w') as f:
    f.write(json.dumps(tree_rules))
