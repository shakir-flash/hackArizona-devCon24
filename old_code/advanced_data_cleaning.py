# advanced_data_cleaning.py

import os
import pandas as pd
import numpy as np
import streamlit as st

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: trim whitespace, lowercase, replace spaces with underscores.
    """
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def advanced_clean_data(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """
    Applies advanced cleaning:
      - Standardize column names.
      - Replace -9999 with NaN.
      - Drop rows that are entirely NaN.
      - Parse DateTime columns (if column name contains 'date' or 'time').
      - Convert temperature values from Fahrenheit to Celsius if max > 50.
      - Clip outliers using the IQR method.
      - Remove duplicate rows.
    """
    df = standardize_columns(df)
    
    # Replace missing value code
    df.replace(-9999, np.nan, inplace=True)
    
    # Drop rows that are entirely NaN (e.g., extra metadata rows)
    df.dropna(how="all", inplace=True)
    
    # Parse datetime columns (if any)
    for col in df.columns:
        if "date" in col or "time" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Temperature conversion: if column contains "temp" and max > 50, convert Fahrenheit to Celsius.
    for col in df.columns:
        if "temp" in col:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].max() > 50:
                df[col] = (df[col] - 32) * 5.0/9.0
    
    # Outlier detection using IQR method for numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df

@st.cache_data
def load_and_advanced_clean_all_csvs(data_folder: str) -> dict:
    """
    Loads all CSV files from the specified folder, applies advanced cleaning,
    and returns a dictionary of cleaned DataFrames keyed by file name.
    """
    cleaned_files = {}
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_folder, file_name)
            try:
                df = pd.read_csv(file_path)
            except Exception:
                continue  # Skip file if error occurs
            cleaned_df = advanced_clean_data(df, file_name)
            cleaned_files[file_name] = cleaned_df
    return cleaned_files
