import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

"""
Phase Two: The preprocessing of the datasets

Data Cleaning: Dropping data that is not needed, handling missing data, duplicated data, and outliers. 

Will use processing techniques for scaling normalization and sampling (random sampling)

Programmed By: Dante Anzalone, Dustin McDonnell, and Leidy Rojas Huisa
Course: CSCI 4105: Knowledge Discovery and Data Mining
Professor: Professor Wei
Date: 2024-10-21
"""

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
def display_data (descriptor, dataset, display, extra_info):
    """
    Accepts a pandas Dataframe and displays it in a formatted table.
    Allows user to give a boolean on whether to show extra information after
    the table is displayed with describe().
    :param descriptor: Description of the data (header)
    :param dataset: Dataset to be displayed, the pandas Dataframe
    :param display: Boolean on whether to display information excluding extra information
    :param extra_info: Yes if user wishes to utilize describe(), no for exclude
    :return: None
    """
    # To display every row and column in the console
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    if display:
        print(f"\n{descriptor} Dataset Information\n\n{dataset}")

    if extra_info:
        print(f"\nExtra Information on {descriptor} \n\n {dataset.describe()}\n\n(Rows, Columns) = {dataset.shape}\n")


def drop_column (dataset, column):
    """
    Accepts a pandas Dataframe and removes the specified column from the dataset.
    :param dataset: Dataset to be modified
    :param column: Dataset's column to be removed
    :return: Modified dataset/Dataframe
    """
    data_copy = dataset.copy()  # Good practice

    # Pandas method: column to remove, axis = 1 specifies column, and inplace meaning modifying DataFrame directly
    data_copy.drop(column, axis = 1, inplace = True)

    return data_copy


# Useful to see where the NaN values are in what column
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


# Dealing with potential duplicates
# Used the following function to count duplicated rows
def count_duplicated_rows (dataset):
    """
    Accepts a pandas Dataframe and counts duplicated rows.
    :param dataset: pandas Dataframe
    :return: Sum of duplicated rows
    """
    return dataset.duplicated().sum()


# Insuring that the data is clean
def drop_duplicates (dataset):
    """
    Accepts a pandas Dataframe and removes duplicated rows.
    :param dataset: pandas Dataframe
    :return: Modified Dataset without duplicated rows
    """
    data_copy = dataset.copy()
    return data_copy.drop_duplicates()


# Dealing with outliers
# First step: Visualization of Dataset using a Histogram
# Outliers may have clinical relevance... e.g., severity in poor MMSE scores
def display_histogram (dataset, title_header, column):
    """
    Accepts a pandas Dataframe and plots a histogram of the frequency of different
    data.
    :param dataset: pandas Dataframe
    :param title_header: Title of the plot
    :param column: Which column to focus on
    :return: None
    """
    bin_size = int(np.ceil(np.sqrt(dataset.shape[0])))

    plt.hist(dataset[column], bins = bin_size)
    plt.title("Histogram of " + title_header)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


# Random Sampling without replacement
def sample_without_replacement (dataset, sample):
    """
    Accepts a pandas Dataframe and samples data without replacement.
    :param dataset: pandas Dataframe
    :param sample: Number of rows to be sampled
    :return: The sampled pandas Dataframe
    """
    data_copy = dataset.copy()
    # frac = Number of rows to be sampled, prevents rows to be selected more than once, and sample along rows
    data_copy = data_copy.sample(n = sample, replace = False, axis = 0)
    return data_copy


def run ():
    # Oasis Longitudinal Demographics and Predictions Datasets
    dataframe_oasis = load_data("oasis_longitudinal_demographics.xlsx")
    dataframe_predictions = load_data("Predictions.xlsx")

    # Dropping columns not useful to our study: Subject ID and MRI ID in Oasis Longitudinal Demographics dataset and Subject ID in Predictions dataset
    dataframe_oasis_modified = drop_column(dataframe_oasis, "Subject ID")
    dataframe_oasis_modified = drop_column(dataframe_oasis_modified, "MRI ID")
    dataframe_predictions_modified = drop_column(dataframe_predictions, "Subject ID")

    # Dropping NaN Rows: SES column had 19 NaN values and MMSE had 2
    print(f"Before Drop: Number of NaNs in Oasis Longitudinal Demographics Dataset\n\n{count_nan(dataframe_oasis)}\n")
    print(f"Before Drop: Number of NaNs in Predictions Dataset\n\n{count_nan(dataframe_predictions)}\n")

    # Note on Oasis: Went from 373 initial rows to 354 rows after drop
    dataframe_oasis_modified = drop_nan_rows(dataframe_oasis_modified)

    print(f"After Drop: Number of NaNs in Oasis Longitudinal Demographics Dataset\n\n{count_nan(dataframe_oasis_modified)}\n")
    # No NaNs in Predictions Dataset

    # Dropping Duplicated Rows
    print(f"Number of Duplicated Rows in Oasis Longitudinal Demographics Dataset: {count_duplicated_rows(dataframe_oasis_modified)}\n")
    print(f"Number of Duplicated Rows in Predictions Dataset: {count_duplicated_rows(dataframe_predictions_modified)}\n")

    dataframe_oasis_modified = drop_duplicates(dataframe_oasis_modified)
    dataframe_predictions_modified = drop_duplicates(dataframe_predictions_modified)

    # Dealing with Outliers
    # Question to consider: Does the Dataset have outliers worth removing?
    # Visualization
    """
    p.display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - Visits", 'Visit')
    p.display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - MR Delays", 'MR Delay')
    p.display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - Age Ranges", 'Age')

    p.display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - EDUC Scores", 'EDUC')
    p.display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - SES Scores", 'SES')
    p.display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - MMSE Scores", 'MMSE')
    p.display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - CDR Scores", 'CDR')
    p.display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - eTIV Scores", 'eTIV')
    p.display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - nWBV Scores", 'nWBV')
    p.display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - ASF Scores", 'ASF')

    p.display_histogram(dataframe_predictions_modified, "Predictions - Ages", 'Age')
    p.display_histogram(dataframe_predictions_modified, "Predictions - CDR Scores", 'CDR')
    p.display_histogram(dataframe_predictions_modified, "Predictions - MMSE Scores", 'MMSE')
    p.display_histogram(dataframe_predictions_modified, "Predictions - MR Delay", 'MR Delay')
    p.display_histogram(dataframe_predictions_modified, "Predictions - SES", 'SES')
    p.display_histogram(dataframe_predictions_modified, "Predictions - Visits", 'Visit')
    p.display_histogram(dataframe_predictions_modified, "Predictions - Nondemented Confidence", 'confidence(Nondemented)')
    p.display_histogram(dataframe_predictions_modified, "Predictions - Demented Confidence", 'confidence(Demented)')
    p.display_histogram(dataframe_predictions_modified, "Predictions - Converted Confidence", 'confidence(Converted)')
    """
    # Displaying Initial Datasets
    display_data("Initial Oasis Longitudinal Demographics", dataframe_oasis, False, False)
    display_data("Initial Predictions", dataframe_predictions, False, False)

    # Displaying Modified Dataset
    display_data("Modified Oasis Longitudinal Demographics", dataframe_oasis_modified, True, True)
    display_data("Modified Predictions", dataframe_predictions_modified, True, True)

    # Creating Sample Datasets
    sample_size_oasis = math.ceil(dataframe_oasis_modified.shape[0] * 0.05)
    sample_size_predictions = math.ceil(dataframe_predictions.shape[0] * 0.05)

    sample_oasis = sample_without_replacement(dataframe_oasis_modified, sample_size_oasis)
    sample_predictions = sample_without_replacement(dataframe_predictions_modified, sample_size_predictions)

    # Displaying Sample Datasets
    display_data("Sample of Oasis Longitudinal Demographics", sample_oasis, False, False)
    display_data("Sample of Predictions", sample_predictions, False, False)

    return dataframe_oasis_modified, dataframe_predictions_modified