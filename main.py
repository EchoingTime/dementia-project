import preprocess as p
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
    # Note on Oasis: Went from 373 initial rows to 354 rows after drop
    dataframe_oasis_modified = p.drop_nan_rows(dataframe_oasis)
    # No NaNs in Predictions Dataset

    # Dropping Duplicated Rows | No Duplicated Rows
    # print(p.count_duplicated_rows(dataframe_oasis_modified))
    # print(p.count_duplicated_rows(dataframe_predictions))

    # Dealing with Outliers

    # Displaying Datasets
    p.display_data("Oasis Longitudinal Demographics", dataframe_oasis_modified, False, True)
    p.display_data("Predictions", dataframe_predictions, False, True)
