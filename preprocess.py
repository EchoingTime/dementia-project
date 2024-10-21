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

# For displaying all rows and columns of the Dataset
def display_data (descriptor, dataset, extra_info):
    """
    Accepts a pandas Dataframe and displays it in a formatted table.
    Allows user to give a boolean on whether to show extra information after
    the table is displayed with describe().
    :param descriptor: Description of the data (header)
    :param dataset: Dataset to be displayed, the pandas Dataframe
    :param extra_info: Yes if user wishes to utilize describe(), no for exclude
    :return: None
    """
    # To display every row and column in the console
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(f"\n{descriptor} Dataset Information\n\n{dataset}")

    if extra_info:
        print(f"\nExtra Information on {descriptor} \n\n {dataset.describe()}\n")
