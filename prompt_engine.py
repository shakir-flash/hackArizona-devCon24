# prompt_engine.py
def build_zone_prompt(dataset_name, df):
    sample = df.head(10).to_string(index=False)
    return f"""
You are a digital scientist managing Biosphere 2 zone: {dataset_name}.
Here is sample sensor data:
{sample}

Analyze this data and suggest:
- Any environmental imbalance
- Possible corrections
- Scientific insights or anomalies
"""
