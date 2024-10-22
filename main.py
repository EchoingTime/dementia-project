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

    # Replacing NaN Rows with Mean
    dataframe_oasis_modified = preprocess.drop_nan_rows(dataframe_oasis)

    # Displaying Datasets
    preprocess.display_data("Oasis Longitudinal Demographics", dataframe_oasis_modified, True)
    # preprocess.display_data("Predictions", dataframe_predictions, True)