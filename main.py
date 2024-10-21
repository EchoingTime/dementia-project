import preprocess

"""
Used to run project's programs

Programmed By: Dante Anzalone, Dustin McDonnell, and Leidy Rojas Huisa
Course: CSCI 4105: Knowledge Discovery and Data Mining
Professor: Professor Wei
Date: 2024-10-21
"""
if __name__ == '__main__':
    # Oasis Longitudinal Demographics and Predictions Datasets
    dataframe_oasis = preprocess.load_data("oasis_longitudinal_demographics.xlsx")
    dataframe_predictions = preprocess.load_data("Predictions.xlsx")

    # Displaying Datasets
    preprocess.display_data("Oasis Longitudinal Demographics", dataframe_oasis, True)
    preprocess.display_data("Predictions", dataframe_predictions, True)
