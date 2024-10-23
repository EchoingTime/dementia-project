import preprocess as p
import normalization as norm
import math

"""
Used to run project's programs

Programmed By: Dante Anzalone, Dustin McDonnell, and Leidy Rojas Huisa
Course: CSCI 4105: Knowledge Discovery and Data Mining
Professor: Professor Wei
Date: 2024-10-21
"""

if __name__ == '__main__':
    # Oasis Longitudinal Demographics and Predictions Datasets
    dataframe_oasis = p.load_data("oasis_longitudinal_demographics.xlsx")
    dataframe_predictions = p.load_data("Predictions.xlsx")

    # Dropping NaN Rows: SES column had 19 NaN values and MMSE had 2
    print(f"Before Drop: Number of NaNs in Oasis Longitudinal Demographics Dataset\n\n{p.count_nan(dataframe_oasis)}\n")
    print(f"Before Drop: Number of NaNs in Predictions Dataset\n\n{p.count_nan(dataframe_predictions)}\n")

    # Note on Oasis: Went from 373 initial rows to 354 rows after drop
    dataframe_oasis_modified = p.drop_nan_rows(dataframe_oasis)

    print(f"After Drop: Number of NaNs in Oasis Longitudinal Demographics Dataset\n\n{p.count_nan(dataframe_oasis_modified)}\n")
    # No NaNs in Predictions Dataset

    # Dropping Duplicated Rows
    print(f"Number of Duplicated Rows in Oasis Longitudinal Demographics Dataset: {p.count_duplicated_rows(dataframe_oasis_modified)}\n")
    print(f"Number of Duplicated Rows in Predictions Dataset: {p.count_duplicated_rows(dataframe_predictions)}\n")

    dataframe_oasis_modified = p.drop_duplicates(dataframe_oasis_modified)
    dataframe_predictions_modified = p.drop_duplicates(dataframe_predictions)

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
    p.display_data("Initial Oasis Longitudinal Demographics", dataframe_oasis, False, False)
    p.display_data("Initial Predictions", dataframe_predictions, False, False)

    # Displaying Modified Dataset
    p.display_data("Modified Oasis Longitudinal Demographics", dataframe_oasis_modified, False, False)
    p.display_data("Modified Predictions", dataframe_predictions_modified, False, False)

    # Creating Sample Datasets
    sample_size_oasis = math.ceil(dataframe_oasis_modified.shape[0] * 0.25)
    sample_size_predictions = math.ceil(dataframe_predictions.shape[0] * 0.25)

    sample_oasis = p.sample_without_replacement(dataframe_oasis_modified, sample_size_oasis)
    sample_predictions = p.sample_without_replacement(dataframe_predictions_modified, sample_size_predictions)

    # Displaying Sample Datasets
    p.display_data("Sample of Oasis Longitudinal Demographics", sample_oasis, False, False)
    p.display_data("Sample of Predictions", sample_predictions, False, False)

    # Normalization of data
    # norm.run(dataframe_oasis_modified)
    # norm.run(dataframe_predictions_modified)