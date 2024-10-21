"""
Phase One: The preprocessing of the datasets

Handling, for data cleaning, missing data, duplicated data, outliers, and zeros.

Will use processing techniques for scaling normalization (standardization and normalization)
and sampling (random sampling)

Programmed By: Dante Anzalone, Dustin McDonnell, and Leidy Rojas Huisa
Course: CSCI 4105: Knowledge Discovery and Data Mining
Professor: Professor Wei
Date: 2024-10-21
"""
# Importing libraries
import pandas as pd

# Importing Dataset via utilizing read_excel
def load_data (file_name):
    """
    Accepts a file and loads the data into a pandas dataframe.
    :param file_name: Name of the file
    :return: Pandas dataframe
    """
    try: # Ensure file is correctly loaded
        excel_dataframe = pd.read_excel (file_name)
        return excel_dataframe
    except Exception as e:
        print(f"There was an error reading the file: {e}")
