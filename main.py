import numpy as np
import preprocess as p
import normalization as norm
import dataMiningModels

"""
Used to run project's programs

Programmed By: Dante Anzalone, Dustin McDonnell, and Leidy Rojas Huisa
Course: CSCI 4105: Knowledge Discovery and Data Mining
Professor: Professor Wei
Date: 2024-10-21
"""

if __name__ == '__main__':
    # Data Cleaning and Preprocessing of data
    dataframe_oasis_modified, dataframe_predictions_modified = p.run()

    # Normalization of data (Need Updating Here and documents for Phase Three)
    # oasis_normalized = norm.run(dataframe_oasis_modified)
    # predictions_normalized = norm.run(dataframe_predictions_modified)

    # Normalization
    oasis_normalized = norm.standard_scale(dataframe_oasis_modified.select_dtypes(include='number').to_numpy())
    predictions_normalized = norm.standard_scale(dataframe_predictions_modified.select_dtypes(include='number').to_numpy())

    # Verifying column names
    print("Columns in dataframe_oasis_modified:", dataframe_oasis_modified.columns)

    # Clustering
    print("\n Clustering Results: ")
    clusters, centroids = dataMiningModels.manual_kmeans(oasis_normalized, k=3)
    dataframe_oasis_modified['Cluster'] = clusters  # Add clusters to the dataframe
    print(f"Cluster Assignments:\n{dataframe_oasis_modified[['Cluster']].value_counts()}")
    print(f"Cluster Centroids:\n{centroids}")

    # Association Rule Mining
    print("\nAssociation Rule Mining: ")
    binary_data = (oasis_normalized > 0).astype(int)  # normalized data to binary for mining
    support = dataMiningModels.calculate_support(binary_data, [0])
    confidence = dataMiningModels.calculate_confidence(binary_data, [0], [1])
    lift = dataMiningModels.calculate_lift(binary_data, [0], [1])
    print(f"Support: {support}, Confidence: {confidence}, Lift: {lift}")

    # Decision Tree (we need to think about the target, 'Group' is just an example)
    print("\nDecision Tree:  ")

    # Extract features
    X = dataframe_oasis_modified.select_dtypes(include=[np.number]).drop(columns=['Group'],errors='ignore').to_numpy()
    # Extract target variable
    y = dataframe_oasis_modified['Group'].to_numpy(dtype=int)  # Can replace '' with any other target column

    # Validate shapes and data types
    print("Feature matrix shape:", X.shape)
    print("Target variable shape:", y.shape)

    tree = dataMiningModels.build_decision_tree(X, y)
    predictions = [dataMiningModels.predict_decision_tree(tree, x) for x in X]
    dataframe_oasis_modified['Predictions'] = predictions
    print(f"Decision Tree Predictions:\n{dataframe_oasis_modified[['Predictions']].value_counts()}")

    # Evaluate the model
    accuracy = np.mean(predictions == y)
    print(f"Decision Tree Accuracy: {accuracy:.2f}")

    # Save the results
    dataframe_oasis_modified.to_csv("oasis_results.csv", index=False)
    dataframe_predictions_modified.to_csv("predictions_results.csv", index=False)
    print("\nResults saved to 'oasis_results.csv' and 'predictions_results.csv'")

