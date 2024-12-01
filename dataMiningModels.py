import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
from collections import defaultdict



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

def apriori(binary_data, min_support=0.1):
    num_transactions = len(binary_data)
    item_counts = defaultdict(int)

    # Count individual items
    for index, transaction in binary_data.iterrows():
        for col in binary_data.columns:
            if transaction[col] == 1:
                item_counts[frozenset([col])] += 1

    # Filter by support
    frequent_itemsets = {item: count / num_transactions for item, count in item_counts.items() if count / num_transactions >= min_support}

    k = 2
    while True:
        # Generate candidate itemsets of size k
        candidates = combinations(frequent_itemsets.keys(), k)
        candidate_counts = defaultdict(int)
        for index, transaction in binary_data.iterrows():
            transaction_items = set([col for col in binary_data.columns if transaction[col] == 1])
            for candidate in candidates:
                if transaction_items.issuperset(set().union(*candidate)):
                    candidate_counts[frozenset().union(*candidate)] += 1

        # Filter candidates by support
        new_frequent_itemsets = {item: count / num_transactions for item, count in candidate_counts.items() if count / num_transactions >= min_support}
        if not new_frequent_itemsets:
            break

        frequent_itemsets.update(new_frequent_itemsets)
        k += 1

    return frequent_itemsets

def extract_rules(frequent_itemsets, min_confidence=0.6):
    """
    Extract association rules from frequent itemsets.
    """
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            for antecedent in combinations(itemset, len(itemset) - 1):
                consequent = itemset.difference(antecedent)
                if consequent:
                    antecedent_support = frequent_itemsets[frozenset(antecedent)]
                    rule_support = frequent_itemsets[itemset]
                    confidence = rule_support / antecedent_support
                    consequent_support = frequent_itemsets[consequent]
                    lift = confidence / consequent_support

                    if confidence >= min_confidence:
                        rules.append({
                            'Antecedent': antecedent,
                            'Consequent': consequent,
                            'Support': rule_support,
                            'Confidence': confidence,
                            'Lift': lift
                        })
    return rules

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


def run (dataframe_oasis_modified, dataframe_predictions_modified,oasis_normalized):
    """
    Runs the data mining techniques and generates decision trees.
    :param dataframe_oasis_modified: Oasis Dataset DataFrame
    :param dataframe_predictions_modified: Predictions Dataset DataFrame
    :param oasis_normalized: Normalized Oasis Dataset DataFrame
    :return: None
    """
    # Verifying column names
    print("Columns in dataframe_oasis_modified:", dataframe_oasis_modified.columns)

    # Clustering
    print("\n Clustering Results: ")
    clusters, centroids = manual_kmeans(oasis_normalized, k=3)
    dataframe_oasis_modified['Cluster'] = clusters  # Add clusters to the dataframe
    print(f"Cluster Assignments:\n{dataframe_oasis_modified[['Cluster']].value_counts()}")
    print(f"Cluster Centroids:\n{centroids}")

    # Risk Factor Analysis
    print("\nCluster Risk Factor Analysis:")
    cluster_summary = dataframe_oasis_modified.groupby('Cluster').mean()
    print(cluster_summary)

    # Visualization: Plot Clusters (Age vs MMSE)
    plt.figure(figsize=(8, 6))
    for cluster in range(3):  # Adjust for the number of clusters
        cluster_data = dataframe_oasis_modified[dataframe_oasis_modified['Cluster'] == cluster]
        plt.scatter(
            cluster_data['Age'], cluster_data['MMSE'], label=f'Cluster {cluster}', alpha=0.6
        )
    plt.title("Cluster Analysis: Age vs MMSE")
    plt.xlabel("Age (Normalized)")
    plt.ylabel("MMSE (Normalized)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Bar Chart: Proportion of Demented/Converted Patients in Each Cluster
    group_proportions = dataframe_oasis_modified.groupby('Cluster')['Group'].value_counts(normalize=True).unstack()
    group_proportions.plot(kind='bar', stacked=True, figsize=(8, 6), alpha=0.8)
    plt.title("Group Proportions by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Proportion")
    plt.legend(title="Group", labels=['Nondemented', 'Demented', 'Converted'])
    plt.grid(axis='y')
    plt.show()

    # Association Rule Mining
    print("\nAssociation Rule Mining:")
    # Refined binary conversion for association rule mining
    dataframe_oasis_modified['MMSE_Risk'] = (dataframe_oasis_modified['MMSE'] < 24).astype(int) # Common clinical threshold for concern
    dataframe_oasis_modified['CDR_Risk'] = (dataframe_oasis_modified['CDR'] > 0.5).astype(int) # Common clinical threshold for CDR
    dataframe_oasis_modified['Age_Risk'] = (dataframe_oasis_modified['Age'] > 70).astype(int) # Commonly used age threshold for 'elderly'

    binary_data = dataframe_oasis_modified[['MMSE_Risk', 'CDR_Risk', 'Age_Risk']]

    # Run Apriori and extract rules
    frequent_itemsets = apriori(binary_data, min_support=0.1)
    rules = extract_rules(frequent_itemsets, min_confidence=0.6)

    # Print results
    print("\nFrequent Itemsets:")
    for itemset, support in frequent_itemsets.items():
        print(f"Itemset: {itemset}, Support: {support:.2f}")

    print("\nAssociation Rules:")
    for rule in rules:
        print(
            f"Rule: {rule['Antecedent']} -> {rule['Consequent']}, Support: {rule['Support']:.2f}, Confidence: {rule['Confidence']:.2f}, Lift: {rule['Lift']:.2f}")

    # Visualization
    if rules:
        feature_labels = ['MMSE_Risk', 'CDR_Risk', 'Age_Risk']  # Adjust based on actual data and rules
        supports = [rule['Support'] for rule in rules]  # Dynamically compute supports
        plt.bar(feature_labels, supports, color='skyblue')
        plt.title("Support of Risk Factors")
        plt.ylabel("Support")
        plt.xlabel("Features")
        plt.ylim(0, 1)
        plt.show()
    else:
        print("No rules met the specified support and confidence thresholds.")


    # Decision Tree (we need to think about the target, 'Group' is just an example)
    print("\nDecision Tree:  ")

    # Extract features
    X = dataframe_oasis_modified.select_dtypes(include=[np.number]).drop(columns=['Group'], errors='ignore').to_numpy()
    # Extract target variable
    y = dataframe_oasis_modified['Group'].to_numpy(dtype=int)  # Can replace '' with any other target column

    # Validate shapes and data types
    print("Feature matrix shape:", X.shape)
    print("Target variable shape:", y.shape)

    # Building tree
    tree = build_decision_tree(X, y)
    predictions = [predict_decision_tree(tree, x) for x in X]
    dataframe_oasis_modified['Predictions'] = predictions
    print(f"Decision Tree Predictions:\n{dataframe_oasis_modified[['Predictions']].value_counts()}")

    # Evaluate the model
    accuracy = np.mean(predictions == y)
    print(f"Decision Tree Accuracy: {accuracy:.2f}")

    # Save the results
    dataframe_oasis_modified.to_csv("oasis_results.csv", index=False)
    dataframe_predictions_modified.to_csv("predictions_results.csv", index=False)
    print("\nResults saved to 'oasis_results.csv' and 'predictions_results.csv'\n")