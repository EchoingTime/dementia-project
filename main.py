import preprocess as p
import normalization as norm
import dataMiningModels
import sys

"""
Used to run project's classes

Programmed By: Dante Anzalone, Dustin McDonnell, and Leidy Rojas Huisa
Course: CSCI 4105: Knowledge Discovery and Data Mining
Professor: Duo Wei
Date: 2024-10-21
"""

if __name__ == '__main__':
    # Data Cleaning and Preprocessing of data
    dataframe_oasis_modified, dataframe_predictions_modified = p.run()

    # Normalization
    oasis_normalized, predictions_normalized = norm.run(dataframe_oasis_modified, dataframe_predictions_modified)

    # Data Mining
    dataMiningModels.run(dataframe_oasis_modified, dataframe_predictions_modified,oasis_normalized)

    sys.exit()