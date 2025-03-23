# prompt_engine.py

def build_zone_prompt(dataset_name, df):
    sample = df.head(10).to_string(index=False)
    return f"""
You are an intelligent environmental agent operating in Biosphere 2.
You are now inside the zone: **{dataset_name}**.

Here is a preview of current sensor readings in this zone:
{sample}

Based on the data above, what balancing actions or insights should you generate to maintain optimal conditions in this zone?

List actionable recommendations as if you are managing this system.
"""
