import numpy as np

"""
Processing technique: Scaling Normalization 

Programmed By: Dante Anzalone, Dustin McDonnell, and Leidy Rojas Huisa
Course: CSCI 4105: Knowledge Discovery and Data Mining
Professor: Professor Wei
Date: 2024-10-22
"""

def min_max(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    normalized_data = (data - data_min) / (data_max - data_min)
    print("Min-Max Normalized Data:\n", normalized_data)


def standard_scale(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    scaled_data = (data - data_mean) / data_std
    return scaled_data


def normalize_scale(data):
    l2_norm = np.linalg.norm(data, axis=1, keepdims=True)
    normalized_data = data / l2_norm
    print("L2 Normalized Data:\n", normalized_data)


def run(data):
    while True:
        choice = input("Enter a number. 1 : Min Max Normalization, 2 : Standard Scale Normalization, "
                       "3 : Euclidean Normalization\n")
        if choice == "1":
            min_max(data)
            break
        elif choice == "2":
            standard_scale(data)
            break
        elif choice == "3":
            normalize_scale(data)
            break
        else:
            "Please enter either 1, 2, or 3."