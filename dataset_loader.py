# dataset_loader.py
import os
import pandas as pd

def load_zone_datasets(folder_path):
    zones = {}
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(folder_path, file))
                df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
                zones[file] = df
            except:
                continue
    return zones
