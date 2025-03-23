# data_utils.py

import scipy.stats as stats

def extended_summarize_df(df, file_name: str) -> str:
    """
    Generates an extended summary of the DataFrame's numeric columns including:
    - Mean, median, standard deviation
    - IQR, skewness, kurtosis
    - Selected percentiles (10th, 25th, 75th, 90th)
    """
    summary_lines = [f"Extended Summary for {file_name}:"]
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        series = df[col].dropna()
        mean_val = series.mean()
        median_val = series.median()
        std_val = series.std()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        iqr_val = Q3 - Q1
        skew_val = stats.skew(series)
        kurt_val = stats.kurtosis(series)
        p10 = series.quantile(0.10)
        p25 = series.quantile(0.25)
        p75 = series.quantile(0.75)
        p90 = series.quantile(0.90)
        summary_lines.append(
            f"{col}: mean={mean_val:.2f}, median={median_val:.2f}, std={std_val:.2f}, IQR={iqr_val:.2f}, skew={skew_val:.2f}, kurtosis={kurt_val:.2f}, p10={p10:.2f}, p25={p25:.2f}, p75={p75:.2f}, p90={p90:.2f}"
        )
    return "\n".join(summary_lines)
