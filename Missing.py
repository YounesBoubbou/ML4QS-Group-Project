import pandas as pd
import numpy as np

training = pd.read_csv("Gaming_Data.csv")


def clean_training(df):
    threshold = 3

    # Initialize an empty DataFrame to store outliers
    outliers = pd.DataFrame()

    # Identify Boolean columns (with 0 and 1 values)
    bool_cols = [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]

    # Initialize an empty DataFrame to store outliers
    outliers = pd.DataFrame()

    # Iterate over each numerical column excluding Boolean columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in bool_cols:
            # Calculate z-scores for the current column
            z_scores = (df[col] - df[col].mean()) / df[col].std()

            # Filter rows with z-scores greater than the threshold
            outliers_col = df[abs(z_scores) > threshold]

            # Add the outliers for the current column to the outliers DataFrame
            outliers = pd.concat([outliers, outliers_col])

    # Remove duplicate rows from the outliers DataFrame (if any)
    outliers = outliers.drop_duplicates()

    df_cleaned = df.drop(outliers.index)

    # Reset index of the cleaned DataFrame
    df_cleaned = df_cleaned.reset_index(drop=True)

    # Integrate the value column


    # Save the cleaned dataset to a new file
    df_cleaned.to_csv('cleaned_gaming_data.csv', index=False)

def clean_testing(df):
    threshold = 3

    # Initialize an empty DataFrame to store outliers
    outliers = pd.DataFrame()

    # Identify Boolean columns (with 0 and 1 values)
    bool_cols = [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]

    # Initialize an empty DataFrame to store outliers
    outliers = pd.DataFrame()

    # Iterate over each numerical column excluding Boolean columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in bool_cols:
            # Calculate z-scores for the current column
            z_scores = (df[col] - df[col].mean()) / df[col].std()

            # Filter rows with z-scores greater than the threshold
            outliers_col = df[abs(z_scores) > threshold]

            # Add the outliers for the current column to the outliers DataFrame
            outliers = pd.concat([outliers, outliers_col])

    # Remove duplicate rows from the outliers DataFrame (if any)
    outliers = outliers.drop_duplicates()

    df_cleaned = df.drop(outliers.index)

    # Reset index of the cleaned DataFrame
    df_cleaned = df_cleaned.reset_index(drop=True)


    # Save the cleaned dataset to a new file
    df_cleaned.to_csv('cleaned_testing_dataset.csv', index=False)



clean_training(training)
#clean_testing(testing)