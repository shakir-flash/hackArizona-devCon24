# ml_trainer.py
import os
import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from constants import MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)

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
    joblib.dump(model, f"{MODEL_DIR}/{zone_name}_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/{zone_name}_scaler.pkl")
    return "✅ Batch model trained"

def predict_zone_model(df, zone_name, target_col):
    try:
        model = joblib.load(f"{MODEL_DIR}/{zone_name}_model.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/{zone_name}_scaler.pkl")
        df = df.dropna().select_dtypes(include=['float64', 'int64'])
        if target_col not in df.columns:
            return None
        X = df.drop(columns=[target_col])
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)
    except:
        return None
