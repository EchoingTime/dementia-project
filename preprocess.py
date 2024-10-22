"""
Phase One: The preprocessing of the datasets

Data Cleaning: Handling missing data, duplicated data, outliers, and zeros.

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
        print(f"\nExtra Information on {descriptor} \n\n {dataset.describe()}\n\n(Rows, Columns) = {dataset.shape}\n")


def count_nan (dataset):
    """
    Accepts a pandas Dataframe and counts NaN values for each column
    :param dataset: pandas Dataframe
    :return: Returns the number of NaN values for each column
    """
    return dataset.isnull().sum()


# (Not Utilized) Dealing with missing values
def replace_nan_with_median (dataset, column):
    """
    Accepts a pandas Dataframe and replaces NaN values with median correlating
    to the specific column.
    :param dataset: pandas Dataframe
    :param column: Name of the column which contains the NaN values
    :return: Modified pandas Dataframe with median values for each NaN
    """
    data_copy = dataset.copy() # Good practice

    # Convertion to numeric (values are currently a type string)
    data_copy[column] = pd.to_numeric(data_copy[column], errors = 'coerce')
    # Will replace NaN values with the median of column values
    data_copy[column] = data_copy[column].fillna(data_copy[column].median())

    return data_copy


# (Utilized) Dealing with missing values
# Used due to the sensitivity of dataset, SES and MMSE
def drop_nan_rows (dataset):
    """
    Accepts a pandas Dataframe and removes rows with NaN values.
    :param dataset: Initial Dataset with missing NaN values
    :return: Modified Dataset with NaN values and their respective rows dropped
    """
    data_copy = dataset.copy() # Generates a new Dataset
    # axis = 0 means wanting to drop rows (axis = 1 is for columns)
    # how = 'any' means drop the row if it contains any NaN values ('all' means drop rows where all values are NaN)
    return data_copy.dropna(axis = 0, how = 'any')