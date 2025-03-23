# dataset_loader.py

import os
import pandas as pd
import numpy as np

def standardize_columns(df):
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def correct_temperature_units(df):
    """
    Automatically convert temperatures to Celsius if values suggest Fahrenheit.
    """
    for col in df.columns:
        if 'temp' in col or 'temperature' in col:
            if pd.api.types.is_numeric_dtype(df[col]):
                max_temp = df[col].max(skipna=True)
                min_temp = df[col].min(skipna=True)
                # Heuristic: if temps > 60, assume Fahrenheit
                if max_temp > 60:
                    df[col] = (df[col] - 32) * 5.0 / 9.0
    return df

def clean_dataset(df):
    df = standardize_columns(df)
    df.replace(-9999, np.nan, inplace=True)
    df.dropna(how='all', inplace=True)

    # Parse datetime columns
    for col in df.columns:
        if 'date' in col or 'time' in col:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    df = correct_temperature_units(df)
    df.drop_duplicates(inplace=True)
    return df

def load_zone_datasets(data_folder):
    """
    Loads each CSV dataset independently and applies cleaning.
    Each dataset is treated as a separate zone.
    """
    zones = {}
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(data_folder, file))
                df = clean_dataset(df)
                zones[file] = df
            except Exception as e:
                print(f"Skipping {file}: {e}")
    return zones
