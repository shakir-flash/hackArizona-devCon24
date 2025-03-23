# ml_utils_simple.py
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def train_and_predict(df, target_col):
    df = df.dropna()
    if target_col not in df.columns or df.shape[0] < 5:
        return None, None

    X = df.drop(columns=[target_col]).select_dtypes(include=['float64', 'int64'])
    y = df[target_col]

    if X.shape[1] == 0:
        return None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)
    return model, preds
