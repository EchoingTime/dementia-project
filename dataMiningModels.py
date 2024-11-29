import numpy as np

"""
Phase Three

Programmed By: Dante Anzalone, Dustin McDonnell, and Leidy Rojas Huisa
Course: CSCI 4105: Knowledge Discovery and Data Mining
Professor: Professor Wei
Date: 2024-11-28
"""

# Clustering Functions.
# Understanding risk factors with data clustering by revealing risk factors among patients in similar clusters,
# and helping to identify the potential progress of dementia.


def manual_kmeans(data, k=3, max_iter=100):
    """KMeans clustering."""
    np.random.seed(42)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for iteration in range(max_iter):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)
        new_centroids = np.array([data[cluster_assignments == i].mean(axis=0) for i in range(k)])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return cluster_assignments, centroids

# Association Rule Mining. Using association rules for patient segmentation
# can encounter existing relationships and patterns among patients with related conditions.


def calculate_support(dataset, itemset):
    """Calculate support for an itemset."""
    return np.mean(np.all(dataset[:, itemset], axis=1))


def calculate_confidence(dataset, antecedent, consequent):
    """Calculate confidence for a rule."""
    support_antecedent = calculate_support(dataset, antecedent)
    support_both = calculate_support(dataset, np.concatenate((antecedent, consequent)))
    return support_both / support_antecedent


def calculate_lift(dataset, antecedent, consequent):
    """Calculate lift for a rule."""
    support_consequent = calculate_support(dataset, consequent)
    confidence = calculate_confidence(dataset, antecedent, consequent)
    return confidence / support_consequent


# Decision Tree Functions. Construct a decision tree
# to categorize data by predicting the possibility of a patient developing dementia based on clinical
# and demographic data.

class DecisionTreeNode:
    """Class for a decision tree node."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def gini_impurity(y):
    """Calculate Gini impurity."""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)


def split_dataset(x, y, feature, threshold):
    """Split dataset based on a feature and threshold."""
    left_mask = x[:, feature] <= threshold
    right_mask = x[:, feature] > threshold
    return x[left_mask], y[left_mask], x[right_mask], y[right_mask]


def build_decision_tree(x, y, depth=0, max_depth=5):
    """Build a decision tree."""
    if depth == max_depth or len(np.unique(y)) == 1:
        return DecisionTreeNode(value=np.argmax(np.bincount(y)))

    best_feature, best_threshold = None, None
    best_gini = float('inf')
    for feature in range(x.shape[1]):
        thresholds = np.unique(x[:, feature])
        for threshold in thresholds:
            _, left_y, _, right_y = split_dataset(x, y, feature, threshold)
            gini = (len(left_y) * gini_impurity(left_y) + len(right_y) * gini_impurity(right_y)) / len(y)
            if gini < best_gini:
                best_gini = gini
                best_feature, best_threshold = feature, threshold

    if best_feature is None:
        return DecisionTreeNode(value=np.argmax(np.bincount(y)))

    left_x, left_y, right_x, right_y = split_dataset(x, y, best_feature, best_threshold)
    left_node = build_decision_tree(left_x, left_y, depth + 1, max_depth)
    right_node = build_decision_tree(right_x, right_y, depth + 1, max_depth)
    return DecisionTreeNode(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)


def predict_decision_tree(tree, x):
    """Predict using a decision tree."""
    if tree.value is not None:
        return tree.value
    if x[tree.feature] <= tree.threshold:
        return predict_decision_tree(tree.left, x)
    else:
        return predict_decision_tree(tree.right, x)
