import numpy as np

"""
Processing Technique: Scaling Normalization 

Programmed By: Dante Anzalone, Dustin McDonnell, and Leidy Rojas Huisa
Course: CSCI 4105: Knowledge Discovery and Data Mining
Professor: Professor Wei
Date: 2024-10-22
"""


# (Not Utilized)
def min_max(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    normalized_data = (data - data_min) / (data_max - data_min)
    print("Min-Max Normalized Data:\n", normalized_data)


# (Utilized)
def standard_scale(data):
    """
    Performs standardization of the dataset
    :param data: pandas DataFrame
    :return: A standardized pandas DataFrame
    """
    # Computing the dataset's mean in the form of a vector
    data_mean = np.mean(data, axis=0)  # axis = 0 represents mean across rows for every given column

    # Computing the dataset's standard deviation in the form of a vector
    data_std = np.std(data, axis=0)  # For finding the variability of the data along each column

    # Takes the original dataset and subtracts it by the mean, aka centering
    # Then divides data by the standard deviation, aka scaling
    scaled_data = (data - data_mean) / data_std

    # Insures every feature/column will equally contribute to later utilized machine learning models
    # E.g. K-means clustering
    return scaled_data


# (Not Utilized)
def normalize_scale(data):
    l2_norm = np.linalg.norm(data, axis=1, keepdims=True)
    normalized_data = data / l2_norm
    print("L2 Normalized Data:\n", normalized_data)


# Where the functions are ran
def run(df_oasis_modified, df_predictions_modified):
    """
    Runs the normalization phase
    :param df_oasis_modified: pandas DataFrame containing oasis data
    :param df_predictions_modified: pandas DataFrame containing prediction data
    :return: Normalized DataFrame/datasets of Modified Oasis and Predictions Datasets
    """
    oasis_normalized = standard_scale(df_oasis_modified.select_dtypes(include='number').to_numpy())
    predictions_normalized = standard_scale(df_predictions_modified.select_dtypes(include='number').to_numpy())

    return oasis_normalized, predictions_normalized
