# prompt_utils.py

def build_descriptive_prompt(input_text: str) -> str:
    """
    Builds a simple descriptive prompt.
    """
    return (
        "Describe the environmental conditions based on the following data:\n\n"
        + input_text
    )

def build_analytical_prompt(input_text: str) -> str:
    """
    Builds a simple analytical prompt.
    """
    return (
        "Analyze the following data for trends and anomalies. Propose possible explanations for any irregularities:\n\n"
        + input_text
    )

def build_predictive_prompt(input_text: str) -> str:
    """
    Builds a simple predictive prompt.
    """
    return (
        "Predict potential environmental changes based on the following data. Suggest interventions if the trends continue:\n\n"
        + input_text
    )

def build_hypothesis_generation_prompt(input_text: str) -> str:
    """
    Builds a simple hypothesis generation prompt.
    """
    return (
        "Generate at least three testable hypotheses regarding the relationships between key environmental variables based on the following data:\n\n"
        + input_text
    )
