"""
Functions for loading, cleaning, transforming, and splitting data.
"""

import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional, List, Union, Dict, Any


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV file from the specified filepath into a Pandas DataFrame.

    Args:
        filepath (str): The complete path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If there's an error parsing the CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The file at '{filepath}' was not found.")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from '{filepath}'. Shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        print(f"Warning: The CSV file at '{filepath}' is empty.")
        return pd.DataFrame() # Return an empty DataFrame
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error parsing CSV file at '{filepath}': {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading '{filepath}': {e}")


def split_data(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.0,  # Default to 0, meaning no separate validation set
    random_state: Optional[int] = 42
    ) -> Tuple[pd.DataFrame, ...]:
    """
    Splits a Pandas DataFrame into training, (optional) validation, and test sets.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        target_column (Optional[str]): The name of the target column to use for
                                        stratified splitting. If None, splitting is not stratified.
        test_size (float): The proportion of the dataset to include in the test split.
                           Should be between 0.0 and 1.0.
        val_size (float): The proportion of the dataset to include in the validation split.
                          This split is taken from the *remaining* data after the test split.
                          Should be between 0.0 and 1.0. If 0.0, no validation set is created.
        random_state (Optional[int]): Controls the randomness of the splitting. Pass an int
                                      for reproducible output across multiple function calls.

    Returns:
        Tuple[pd.DataFrame, ...]: A tuple of DataFrames:
                                  (train_df, test_df) if val_size is 0.0,
                                  (train_df, val_df, test_df) if val_size > 0.0.

    Raises:
        ValueError: If `test_size` or `val_size` are out of range, or if `target_column`
                    is provided but not found in the DataFrame.
        Exception: For other unexpected issues during splitting.
    """
    if not (0.0 <= test_size < 1.0):
        raise ValueError("test_size must be between 0.0 and less than 1.0")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be between 0.0 and less than 1.0")
    if test_size + val_size >= 1.0:
        raise ValueError("The sum of test_size and val_size must be less than 1.0")

    stratify_by = None
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        stratify_by = df[target_column]
        print(f"Splitting data with stratification based on column: '{target_column}'")
    else:
        print("Splitting data without stratification.")

    try:
        # First split: Separate out the test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_by
        )

        print(f"Initial split: Train+Val shape: {train_val_df.shape}, Test shape: {test_df.shape}")

        if val_size > 0.0:
            # Calculate validation size relative to the remaining train_val_df
            # val_size_adjusted = val_size / (1 - test_size)
            # A more robust way to handle this without complex math for simple splits:
            # Just perform the split directly on train_val_df
            
            # Re-calculating stratification for the second split, as stratify_by refers to original df
            # If stratified split is needed for val_set as well:
            stratify_for_val = train_val_df[target_column] if target_column else None

            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size, # This is the proportion of train_val_df
                random_state=random_state,
                stratify=stratify_for_val
            )
            print(f"Second split: Train shape: {train_df.shape}, Validation shape: {val_df.shape}")
            return train_df, val_df, test_df
        else:
            # If no validation set is requested, train_val_df is simply the train_df
            train_df = train_val_df
            print(f"Final split: Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            return train_df, test_df

    except Exception as e:
        raise Exception(f"An error occurred during data splitting: {e}")


def process_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Processes the features and target column of a DataFrame.

    This includes:
    1. Converting the boolean target column to numerical (0 or 1).
    2. Applying StandardScaler to numeric features.
    3. Applying OneHotEncoder to string (categorical) features.
    4. Handling potential missing values by imputing (e.g., mean for numeric,
       most frequent for categorical, or by letting the transformers handle NaNs based on strategy).
       Note: sklearn's transformers by default will raise errors for NaNs,
       so a simple imputation step (like fillna) before transformation is good practice if NaNs are expected.
       For simplicity here, we assume basic NaNs are handled by the pipeline or data is clean,
       but in production, more robust imputation might be needed.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        feature_columns (List[str]): A list of column names that are features.
        target_column (str): The name of the target column (boolean type).
        numeric_cols (List[str]): A list of column names that are numeric features.
        categorical_cols (List[str]): A list of column names that are string/categorical features.

    Returns:
        Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
            - X_processed (pd.DataFrame): The DataFrame with processed features.
            - y_processed (pd.Series): The Series with processed target (0 or 1).
            - preprocessor (ColumnTransformer): The fitted preprocessor object, useful for
              transforming new data (e.g., test set).

    Raises:
        ValueError: If any specified feature or target column is not found in the DataFrame,
                    or if feature_columns list does not cover all identified numeric and categorical columns.
    """
    # Validate column existence
    all_specified_columns = feature_columns + [target_column]
    missing_cols = [col for col in all_specified_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following specified columns were not found in the DataFrame: {missing_cols}")

    # Validate that all features are covered by numeric_cols or categorical_cols
    identified_features = set(numeric_cols + categorical_cols)
    if set(feature_columns) != identified_features:
        raise ValueError(
            "The union of 'numeric_cols' and 'categorical_cols' must exactly match 'feature_columns'."
            f" Features specified: {set(feature_columns)}, Identified by type: {identified_features}"
        )

    # Separate features (X) and target (y)
    X = df[feature_columns]
    y = df[target_column].copy() # Use .copy() to avoid SettingWithCopyWarning

    # 1. Convert boolean target to numeric (0 or 1)
    if y.dtype == 'bool':
        y_processed = y.astype(int)
        print(f"Target column '{target_column}' converted from boolean to integer (0/1).")
    elif y.dtype in ['int64', 'float64'] and y.nunique() <= 2: # Assuming 2 classes for boolean-like
        print(f"Target column '{target_column}' is already numeric. Assuming 0/1 encoding.")
        y_processed = y
    else:
        # Handle cases where target might be 'Yes'/'No' strings or other forms
        # For simplicity, if not bool, assume it's already numerical or requires manual handling
        # A more robust solution might include LabelEncoder here if expecting string categories.
        print(f"Warning: Target column '{target_column}' is of type {y.dtype} and not boolean. "
              "Proceeding without explicit conversion for target. Ensure it's 0/1 encoded.")
        y_processed = y


    # 2 & 3. Create preprocessing pipelines for numeric and categorical features
    # Numerical pipeline: Impute with mean (if NaNs present) then scale
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()) # Scales features to have mean 0 and variance 1
    ])

    # Categorical pipeline: Impute with most frequent (if NaNs present) then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for DataFrame output
    ])

    # Create a preprocessor using ColumnTransformer
    # This applies different transformers to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep other columns (if any) as they are
    )

    # Fit the preprocessor and transform the features
    print("Fitting and transforming features...")
    X_processed_array = preprocessor.fit_transform(X)

    # Get feature names after transformation for DataFrame reconstruction
    # This is a bit more complex with ColumnTransformer and OneHotEncoder
    try:
        # For sklearn >= 0.23, get_feature_names_out is available
        feature_names_out = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions or if get_feature_names_out is not enough
        # Manually construct names:
        num_feature_names = numeric_cols
        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
        feature_names_out = list(num_feature_names) + list(cat_feature_names)
        
    X_processed = pd.DataFrame(X_processed_array, columns=feature_names_out, index=X.index)

    print(f"Features processed. Original shape: {X.shape}, Processed shape: {X_processed.shape}")

    return X_processed, y_processed, preprocessor