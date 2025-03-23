# analysis_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

def local_descriptive_analysis(df: pd.DataFrame, file_name: str):
    """
    Performs descriptive analysis:
    - Generates a text summary of numeric columns.
    - Plots histograms for each numeric column.
    Returns the text summary.
    """
    summary_lines = [f"Descriptive Analysis for {file_name}:"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stats = df[col].describe()
        summary_lines.append(
            f"{col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, std={stats['std']:.2f}"
        )
    summary_text = "\n".join(summary_lines)
    
    for col in numeric_cols:
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    return summary_text

def local_analytical_analysis(df: pd.DataFrame, file_name: str, z_thresh=3.0):
    """
    Performs analytical analysis using Z-score based anomaly detection.
    Returns a text summary of potential outliers for each numeric column and plots boxplots.
    """
    outlier_info = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        z_scores = (series - series.mean()) / series.std()
        outliers = series[abs(z_scores) > z_thresh]
        outlier_info[col] = outliers.tolist()
        
        fig, ax = plt.subplots()
        ax.boxplot(series, vert=False)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)
    
    analysis_text = f"Analytical Analysis for {file_name}:\n"
    for col, outliers in outlier_info.items():
        analysis_text += f"{col}: detected {len(outliers)} outliers. Outliers: {outliers}\n"
    return analysis_text

def local_predictive_analysis(df: pd.DataFrame, file_name: str, target_var: str, forecast_steps=10):
    """
    Performs simple time series forecasting using ARIMA on the target variable.
    Plots the forecast and returns a forecast summary.
    """
    if target_var not in df.columns:
        return f"Target variable '{target_var}' not found in {file_name}."
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        else:
            return "No datetime index available for forecasting."
    
    series = df[target_var].dropna()
    if len(series) < 20:
        return "Not enough data for forecasting."
    
    try:
        model = ARIMA(series, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)
        
        fig, ax = plt.subplots()
        series.plot(ax=ax, label="Historical")
        forecast.plot(ax=ax, label="Forecast", style='--', color='red')
        ax.set_title(f"Forecast of {target_var} for next {forecast_steps} periods")
        ax.legend()
        st.pyplot(fig)
        
        return forecast.to_string()
    except Exception as e:
        return f"Error in forecasting: {str(e)}"

def local_hypothesis_generation(df: pd.DataFrame, file_name: str, corr_threshold=0.7):
    """
    Generates rule-based hypotheses by computing the correlation matrix.
    Returns hypotheses if any pair of numeric variables has correlation above the threshold.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return "No numeric data available for hypothesis generation."
    
    corr_matrix = numeric_df.corr()
    hypotheses = []
    for i, col in enumerate(corr_matrix.columns):
        for j in range(i+1, len(corr_matrix.columns)):
            other_col = corr_matrix.columns[j]
            corr_value = corr_matrix.loc[col, other_col]
            if abs(corr_value) >= corr_threshold:
                direction = "positively" if corr_value > 0 else "negatively"
                hypotheses.append(
                    f"Hypothesis: {col} is {direction} correlated with {other_col} (correlation = {corr_value:.2f})."
                )
    if not hypotheses:
        return "No strong correlations detected to form hypotheses."
    return "\n".join(hypotheses)
