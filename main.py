import preprocess as p
import normalization as norm

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
    norm.run(dataframe_oasis_modified)
    norm.run(dataframe_predictions_modified)