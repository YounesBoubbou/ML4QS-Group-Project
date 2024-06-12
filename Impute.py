import pandas as pd
from sklearn.impute import SimpleImputer

# Set file paths
origin_training_set_path = 'Gaming_Data.csv'
imputed_training_set_path = 'imputed_Gaming_Data1.csv'


# Function to impute missing values in the DataFrame
def impute_training_set(df, method='median'):
    # Print the initial columns and their data types
    print("Initial columns and data types:\n", df.dtypes)

    # Remove columns with more than 50% missing values
    missing_ratios = df.isnull().mean()
    df = df.loc[:, missing_ratios <= 0.5]

    # Print the columns and their data types after removing columns with >50% missing values
    print("Columns after removing those with >50% missing values:\n", df.dtypes)

    if method == 'median':
        # Use median to impute missing values
        imputer = SimpleImputer(strategy='median')
        # Select numerical columns
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns

        # Print the numerical columns selected
        print("Numerical columns selected for imputation:\n", num_cols)

        if len(num_cols) == 0:
            raise ValueError("No numerical columns available for imputation.")

        # Check for empty DataFrame after column removal
        if df[num_cols].isnull().all().all():
            raise ValueError("All selected numerical columns contain only NaN values.")

        # Impute missing values
        df[num_cols] = imputer.fit_transform(df[num_cols])
    elif method == 'linear':
        # Use linear interpolation to impute missing values
        df = df.interpolate(method='linear', limit_direction='both')
    else:
        raise ValueError("Invalid imputation method. Choose 'median' or 'linear'.")

    return df


# Read the CSV file into a DataFrame with proper delimiter
df = pd.read_csv(origin_training_set_path, delimiter='\t')

# Check if DataFrame is empty
if df.empty:
    raise ValueError("The input CSV file is empty or not properly loaded.")

# Impute missing values
imputed_df = impute_training_set(df, method='median')

# Save the imputed DataFrame to a new CSV file
imputed_df.to_csv(imputed_training_set_path, index=False)

print(f"Imputed data saved to {imputed_training_set_path}")