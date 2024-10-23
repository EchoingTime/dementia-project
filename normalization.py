import sklearn.preprocessing as sk

"""
Processing technique: Scaling Normalization 

Programmed By: Dante Anzalone, Dustin McDonnell, and Leidy Rojas Huisa
Course: CSCI 4105: Knowledge Discovery and Data Mining
Professor: Professor Wei
Date: 2024-10-22
"""

def min_max(data):
    scaler = sk.MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    print(normalized_data)


def standard_scale(data):
    scaler = sk.StandardScaler()
    normalized_data = scaler.fit_transform(data)
    print(normalized_data)


def normalize_scale(data):
    scaler = sk.Normalizer()
    normalized_data = scaler.fit_transform(data)
    print(normalized_data)


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