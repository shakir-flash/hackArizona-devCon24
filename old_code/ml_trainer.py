# ml_trainer.py
import os
import joblib
import re
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from constants import MODEL_DIR

# ✅ Get absolute path to models directory
ABS_MODEL_DIR = os.path.abspath(MODEL_DIR)
if not os.path.exists(ABS_MODEL_DIR):
    os.makedirs(ABS_MODEL_DIR)

# ✅ Utility to sanitize filenames
def sanitize_filename(name):
    name = name.replace(".csv", "")
    name = re.sub(r'[\\/*?:"<>|]', "_", name)  # Windows-safe
    return name

def train_zone_model(df, zone_name, target_col):
    df = df.dropna().select_dtypes(include=['float64', 'int64'])
    if target_col not in df.columns:
        return "❌ Target not found"

    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SGDRegressor()
    model.fit(X_scaled, y)

    zone_name_clean = sanitize_filename(zone_name)
    model_path = os.path.join(ABS_MODEL_DIR, f"{zone_name_clean}_model.pkl")
    scaler_path = os.path.join(ABS_MODEL_DIR, f"{zone_name_clean}_scaler.pkl")

    # ✅ Ensure directory exists again before saving (safeguard)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, 'wb') as f:
        joblib.dump(model, f)

    with open(scaler_path, 'wb') as f:
        joblib.dump(scaler, f)

    return f"✅ Model trained and saved for {zone_name_clean}"

def predict_zone_model(df, zone_name, target_col):
    try:
        zone_name_clean = sanitize_filename(zone_name)
        model_path = os.path.join(ABS_MODEL_DIR, f"{zone_name_clean}_model.pkl")
        scaler_path = os.path.join(ABS_MODEL_DIR, f"{zone_name_clean}_scaler.pkl")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        df = df.dropna().select_dtypes(include=['float64', 'int64'])
        if target_col not in df.columns:
            return None
        X = df.drop(columns=[target_col])
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)
    except Exception as e:
        return f"❌ Prediction error: {str(e)}"
