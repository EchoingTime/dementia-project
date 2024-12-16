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
Professor: Duo Wei
Date: 2024-10-21
"""

# Importing Dataset via utilizing read_excel
def load_data(file_name):
    """
    Accepts a file and loads the data into a pandas dataframe.
    :param file_name: Name of the file
    :return: Pandas dataframe
    """
    try:  # Ensure file is correctly loaded
        excel_dataframe = pd.read_excel(file_name)
        return excel_dataframe
    except Exception as e:
        print(f"There was an error reading the file: {e}")


# For displaying all rows and columns of the Dataset
def display_data(descriptor, dataset, display, extra_info):
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
        print(f"{descriptor} Dataset Information\n\n{dataset}")
    else:
        print(f"{descriptor} Dataset: Display = FALSE\n")

    if extra_info:
        print(f"\nExtra Information on {descriptor} \n\n {dataset.describe()}\n\n(Rows, Columns) = {dataset.shape}\n")
    else:
        print(f"\n{descriptor} Dataset: Display Extra Information = FALSE\n")


# Dropping Columns Deemed Unuseful
def drop_column(dataset, column):
    """
    Accepts a pandas Dataframe and removes the specified column from the dataset.
    :param dataset: Dataset to be modified
    :param column: Dataset's column to be removed
    :return: Modified dataset/Dataframe
    """
    data_copy = dataset.copy()  # Good practice

    # Pandas method: column to remove, axis = 1 specifies column, and inplace meaning modifying DataFrame directly
    data_copy.drop(column, axis=1, inplace=True)

    return data_copy


# Removes the first 5 characters and then converts the rest to int
def convert_subject_id(subject_id):
    """
    Removes the first 5 characters and then converts the rest to int.
    :param subject_id: Specific Subject ID to Convert
    :return: Converted Subject ID
    """
    return int(subject_id[5:])


# Rounding values for easier viewing
def round_values(value):
    """
    Rounds the value to 4 decimal places.
    :param value: Value to be rounded
    :return: The rounded value
    """
    return round(value, 4)


# Useful to see where the NaN values are in what column
def count_nan(dataset):
    """
    Accepts a pandas Dataframe and counts NaN values for each column
    :param dataset: pandas Dataframe
    :return: Returns the number of NaN values for each column
    """
    return dataset.isnull().sum()


# (Not Utilized) Dealing with missing values
def replace_nan_with_median(dataset, column):
    """
    Accepts a pandas Dataframe and replaces NaN values with median correlating
    to the specific column.
    :param dataset: pandas Dataframe
    :param column: Name of the column which contains the NaN values
    :return: Modified pandas Dataframe with median values for each NaN
    """
    data_copy = dataset.copy()  # Good practice

    # Conversion to numeric (values are currently a type string)
    data_copy[column] = pd.to_numeric(data_copy[column], errors='coerce')
    # Will replace NaN values with the median of column values
    data_copy[column] = data_copy[column].fillna(data_copy[column].median())

    return data_copy


# (Utilized) Dealing with missing values
# Used due to the sensitivity of dataset, SES and MMSE
def drop_nan_rows(dataset):
    """
    Accepts a pandas Dataframe and removes rows with NaN values.
    :param dataset: Initial Dataset with missing NaN values
    :return: Modified Dataset with NaN values and their respective rows dropped
    """
    data_copy = dataset.copy()  # Generates a new Dataset
    # axis = 0 means wanting to drop rows (axis = 1 is for columns)
    # how = 'any' means drop the row if it contains any NaN values ('all' means drop rows where all values are NaN)
    return data_copy.dropna(axis=0, how='any')


# Dealing with potential duplicates
# Used the following function to count duplicated rows
def count_duplicated_rows(dataset):
    """
    Accepts a pandas Dataframe and counts duplicated rows.
    :param dataset: pandas Dataframe
    :return: Sum of duplicated rows
    """
    return dataset.duplicated().sum()


# Insuring that the data is clean
def drop_duplicates(dataset):
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
def display_histogram(dataset, title_header, column):
    """
    Accepts a pandas Dataframe and plots a histogram of the frequency of different
    data.
    :param dataset: pandas Dataframe
    :param title_header: Title of the plot
    :param column: Which column to focus on
    :return: None
    """
    bin_size = int(np.ceil(np.sqrt(dataset.shape[0])))

    plt.hist(dataset[column], bins=bin_size)
    plt.title("Histogram of " + title_header)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


# Random Sampling without replacement
def sample_without_replacement(dataset, sample):
    """
    Accepts a pandas Dataframe and samples data without replacement.
    :param dataset: pandas Dataframe
    :param sample: Number of rows to be sampled
    :return: The sampled pandas Dataframe
    """
    data_copy = dataset.copy()
    # frac = Number of rows to be sampled, prevents rows to be selected more than once, and sample along rows
    data_copy = data_copy.sample(n=sample, replace=False, axis=0)
    return data_copy


# Where the program is run to preprocess and clean the datasets
# There is no function for categorical to numerical values --> It is taken care of in the run function for simplicity
def run():
    """
    Runs the preprocessing pipeline!
    :return: A modified Oasis Longitudinal Demographics Dataset and a Predictions Dataset
    """

    print(f"\nPreprocessing Data\n")

    print(f"-----------------------------------------------------------------------------------------------------\n")

    # Oasis Longitudinal Demographics and Predictions Datasets
    dataframe_oasis = load_data("oasis_longitudinal_demographics.xlsx")
    dataframe_predictions = load_data("Predictions.xlsx")

    print(f"Dropping MRI ID and Hand Columns in Oasis Longitudinal Demographics Dataset.")

    # Dropping columns not useful to our study:
    # MRI ID and Hand in Oasis Longitudinal Demographics dataset
    dataframe_oasis_modified = drop_column(dataframe_oasis, "MRI ID")
    dataframe_oasis_modified = drop_column(dataframe_oasis_modified, "Hand")

    dataframe_predictions_modified = dataframe_predictions.copy()

    # Handling future error given: Set option to opt into the new replace behavior
    pd.set_option('future.no_silent_downcasting', True)

    print(f"\n-----------------------------------------------------------------------------------------------------\n")

    # Converting categorical columns to numerical
    #   Oasis Longitudinal Demographics Dataset
    #       Group Column --> New Representations
    #               Nondemented         2
    #               Demented            1
    #               Converted           0
    #       M/F
    #               M (Male)            1
    #               F (Female)          0
    #       Subject ID dropping first 5 characters and leading zeros
    #   Predictions Dataset
    #       Group
    #               Nondemented         2
    #               Demented            1
    #               Converted           0
    #       M/F
    #               M (Male)            1
    #               F (Female)          0
    #       prediction(Group)
    #               Nondemented         2
    #               Demented            1
    #               Converted           0
    #       Subject ID dropping first 5 characters and leading zeros

    print(f"Converting categorical columns to numerical...\n\n"
          f"Dataframe Oasis Columns Conversion:\n\n"
          f"Group: Nondemented as 2, Demented as 1, Converted as 0\\nn"
          f"M/F: M as 1, F as 0\n\n"
          f"Subject ID: Applying Conversion to Subject IDs, dropping non-integers and then converting to ints\n\n"
          f"eTIV, nWBV, and ASF: Rounding 4 decimal places\n\n"
          f"Dataframe Predictions Columns Conversion:\n\n"
          f"Group: Nondemented as 2, Demented as 1, Converted as 0\n\n"
          f"prediction(Group): Nondemented as 2, Demented as 1, Converted as 0\n\n"
          f"M/F: M as 1, F as 0\n\n"
          f"Subject ID: Applying Conversion to Subject IDs, dropping non-integers and then converting to ints\n\n"
          f"confidence(Nondemented), confidence(Demented), and confidence(Converted): Rounding 4 decimal places\n")

    dataframe_oasis_modified['Group'] = dataframe_oasis_modified['Group'].replace(
        {'Nondemented': 2, 'Demented': 1, 'Converted': 0})
    dataframe_oasis_modified['M/F'] = dataframe_oasis_modified['M/F'].replace({'M': 1, 'F': 0})
    dataframe_oasis_modified['Subject ID'] = dataframe_oasis_modified['Subject ID'].apply(convert_subject_id)
    dataframe_oasis_modified['eTIV'] = dataframe_oasis_modified['eTIV'].apply(round_values)
    dataframe_oasis_modified['nWBV'] = dataframe_oasis_modified['nWBV'].apply(round_values)
    dataframe_oasis_modified['ASF'] = dataframe_oasis_modified['ASF'].apply(round_values)

    dataframe_predictions_modified['Group'] = dataframe_predictions_modified['Group'].replace(
        {'Nondemented': 2, 'Demented': 1, 'Converted': 0})
    dataframe_predictions_modified['prediction(Group)'] = dataframe_predictions_modified['prediction(Group)'].replace(
        {'Nondemented': 2, 'Demented': 1, 'Converted': 0})
    dataframe_predictions_modified['M/F'] = (
        dataframe_predictions_modified['M/F'].replace({'M': 1, 'F': 0}))
    dataframe_predictions_modified['Subject ID'] = (
        dataframe_predictions_modified['Subject ID'].apply(convert_subject_id))
    dataframe_predictions_modified['confidence(Nondemented)'] = (
        dataframe_predictions_modified['confidence(Nondemented)'].apply(round_values))
    dataframe_predictions_modified['confidence(Demented)'] = (
        dataframe_predictions_modified['confidence(Demented)'].apply(round_values))
    dataframe_predictions_modified['confidence(Converted)'] = (
        dataframe_predictions_modified['confidence(Converted)'].apply(round_values))

    print(f"-----------------------------------------------------------------------------------------------------\n")

    # Dropping NaN Rows: SES column had 19 NaN values and MMSE had 2
    print(
        f"Before NaN Drop: Number of NaNs in Oasis Longitudinal Demographics Dataset\n\n{count_nan(dataframe_oasis)}\n")
    print(f"Before NaN Drop: Number of NaNs in Predictions Dataset\n\n{count_nan(dataframe_predictions)}\n")

    # Note on Oasis: Went from 373 initial rows to 354 rows after drop
    dataframe_oasis_modified = drop_nan_rows(dataframe_oasis_modified)

    print(
        f"After NaN Drop: Number of NaNs in Oasis Longitudinal Demographics Dataset\n"
        f"\n{count_nan(dataframe_oasis_modified)}\n"
        f"\nAfter NaN Drop: Number of NaNs in Predictions Dataset\n\n{count_nan(dataframe_predictions)}\n")
    # No NaNs in Predictions Dataset

    print(f"-----------------------------------------------------------------------------------------------------\n")

    # Dropping Duplicated Rows
    print(
        f"Before Duplication Drop: Number of Duplicated Rows in Oasis Longitudinal Demographics Dataset: "
        f"{count_duplicated_rows(dataframe_oasis_modified)}\n")
    print(
        f"Before Duplication Drop: Number of Duplicated Rows in Predictions Dataset: "
        f"{count_duplicated_rows(dataframe_predictions)}\n")

    dataframe_oasis_modified = drop_duplicates(dataframe_oasis_modified)
    dataframe_predictions_modified = drop_duplicates(dataframe_predictions_modified)

    print(
        f"After Duplication Drop: No Duplications in Oasis Longitudinal Demographics Dataset: "
        f"{count_duplicated_rows(dataframe_oasis_modified)}\n"
        f"\nAfter Duplication Drop: Number of Duplicated Rows in Predictions Dataset: "
        f"{count_duplicated_rows(dataframe_predictions_modified)}")

    print(f"\n-----------------------------------------------------------------------------------------------------\n")

    print(f"Generating Visuals to spot Outliers and Reveal Data Patterns (Decided not to remove outliers)")

    # Dealing with Outliers
    # Question to consider: Does the Dataset have outliers worth removing?
    # Visualization
    """
    display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - Groups", 'Group')
    display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - Visits", 'Visit')
    display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - MR Delays", 'MR Delay')
    display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - Male Vs Female", 'M/F')
    display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - Age Ranges", 'Age')

    display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - EDUC Scores", 'EDUC')
    display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - SES Scores", 'SES')
    display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - MMSE Scores", 'MMSE')
    display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - CDR Scores", 'CDR')
    display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - eTIV Scores", 'eTIV')
    display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - nWBV Scores", 'nWBV')
    display_histogram(dataframe_oasis_modified, "Oasis Longitudinal Demographics - ASF Scores", 'ASF')

    display_histogram(dataframe_predictions_modified, "Predictions - Ages", 'Age')
    display_histogram(dataframe_predictions_modified, "Predictions - CDR Scores", 'CDR')
    display_histogram(dataframe_predictions_modified, "Predictions - Male Vs Female", 'M/F')
    display_histogram(dataframe_predictions_modified, "Predictions - MMSE Scores", 'MMSE')
    display_histogram(dataframe_predictions_modified, "Predictions - MR Delay", 'MR Delay')
    display_histogram(dataframe_predictions_modified, "Predictions - SES", 'SES')
    display_histogram(dataframe_predictions_modified, "Predictions - Visits", 'Visit')
    display_histogram(dataframe_predictions_modified, "Predictions - Groups", 'Group')
    
    display_histogram(dataframe_predictions_modified, "Predictions - Nondemented Confidence", 'confidence(Nondemented)')
    display_histogram(dataframe_predictions_modified, "Predictions - Demented Confidence", 'confidence(Demented)')
    display_histogram(dataframe_predictions_modified, "Predictions - Converted Confidence", 'confidence(Converted)')
    display_histogram(dataframe_predictions_modified, "Predictions - Predictions (Groups)", 'prediction(Group)')
    """

    print(f"\n-----------------------------------------------------------------------------------------------------"
          f"\n\nDisplaying Initial Datasets\n")

    # Displaying Initial Datasets
    display_data("Initial Oasis Longitudinal Demographics", dataframe_oasis, False, False)
    display_data("Initial Predictions", dataframe_predictions, False, False)

    print(f"-----------------------------------------------------------------------------------------------------"
          f"\n\nDisplaying Modified Dataset\n")

    # Displaying Modified Dataset
    display_data("Modified Oasis Longitudinal Demographics",
                 dataframe_oasis_modified, False, False)
    display_data("Modified Predictions", dataframe_predictions_modified, False, False)

    print(f"-----------------------------------------------------------------------------------------------------\n")

    print(f"Displaying Sample Datasets\n")

    # Creating Sample Datasets
    sample_size_oasis = math.ceil(dataframe_oasis_modified.shape[0] * 0.05)
    sample_size_predictions = math.ceil(dataframe_predictions.shape[0] * 0.05)

    sample_oasis = sample_without_replacement(dataframe_oasis_modified, sample_size_oasis)
    sample_predictions = sample_without_replacement(dataframe_predictions_modified, sample_size_predictions)

    # Displaying Sample Datasets
    display_data("Sample of Oasis Longitudinal Demographics", sample_oasis, False, False)
    display_data("Sample of Predictions", sample_predictions, False, False)

    return dataframe_oasis_modified, dataframe_predictions_modified