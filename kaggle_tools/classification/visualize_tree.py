# Inspired from the following sklearn example:
# http://scikit-learn.org/stable/modules/tree.html#tree

from IPython.display import Image
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO

import pydotplus  # Using this version of pydot since I am running Python 3.4


def visualize_tree(clf, feature_names, class_names, output_file,
                   method='pdf'):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True,
                         impurity=False)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    if method == 'pdf':
        graph.write_pdf(output_file + ".pdf")
    elif method == 'inline':
        Image(graph.create_png())

    return graph

# An example using the iris dataset

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)


tree_graph = visualize_tree(clf, iris.feature_names,
                            iris.target_names,
                            'classification/iris_decision_tree')
