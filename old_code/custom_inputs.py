# custom_inputs.py

def get_dataset_custom_context(df, file_name: str) -> str:
    """
    Inspects the DataFrame's columns and returns dataset-specific context.
    """
    columns = df.columns.tolist()
    context_lines = [f"Dataset '{file_name}' includes the following variables: {', '.join(columns)}."]
    
    # Adjust context based on common ecological variables:
    if any(col in columns for col in ['temp', 'temperature']):
        context_lines.append("Temperature data is critical; consider diurnal patterns and verify unit conversion.")
    if any(col in columns for col in ['rh', 'relative_humidity']):
        context_lines.append("Relative humidity data influences evaporation and plant stress.")
    if any(col in columns for col in ['ph']):
        context_lines.append("pH values help assess water quality in aquatic ecosystems.")
    if any(col in columns for col in ['salinity']):
        context_lines.append("Salinity levels are important for marine or estuarine conditions.")
    if any(col in columns for col in ['co2']):
        context_lines.append("CO₂ data reflects carbon cycling and greenhouse gas dynamics.")
    
    return "\n".join(context_lines)

def customize_descriptive_input(summary: str, custom_context: str) -> str:
    """
    Enhance the descriptive input with dataset-specific context.
    """
    return (
        "Enhance the following summary with details on data distribution (mean, median, variance) and suggest visualizations (e.g., histograms, boxplots). "
        "Consider the dataset-specific context below:\n\n" 
        + custom_context + "\n\n" + summary
    )

def customize_analytical_input(summary: str, custom_context: str) -> str:
    """
    Enhance the analytical input with instructions for anomaly detection and hypothesis testing, plus dataset context.
    """
    return (
        "Analyze the following summary using techniques such as Isolation Forest or Z-score analysis for anomaly detection, "
        "and perform hypothesis testing (e.g., t-test, ANOVA) on the data trends. Consider the following dataset-specific context:\n\n" 
        + custom_context + "\n\n" + summary
    )

def customize_predictive_input(summary: str, custom_context: str) -> str:
    """
    Enhance the predictive input with time series forecasting context and dataset-specific details.
    """
    return (
        "Using time series forecasting methods (e.g., ARIMA, LSTM), predict future trends based on the following summary. "
        "Factor in the following dataset-specific context:\n\n"
        + custom_context + "\n\n" + summary
    )

def customize_hypothesis_generation_input(summary: str, custom_context: str) -> str:
    """
    Enhance the hypothesis generation input with causal inference instructions and dataset-specific context.
    """
    return (
        "Based on the following summary, use causal inference principles to generate testable hypotheses. "
        "Incorporate the dataset-specific context below to tailor your hypotheses:\n\n"
        + custom_context + "\n\n" + summary
    )
