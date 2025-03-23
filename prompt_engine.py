# prompt_engine.py

def build_prompt(zone, df):
    return f"""
You are a scientific ecosystem expert for Biosphere 2. Analyze the sensor data from zone: {zone} and give insights.

Data Snapshot:
{df.head(10).to_string(index=False)}
"""

def build_small_talk_prompt(zone, summary):
    return f"""
You are a friendly Assistant AI in Biosphere 2 working with your scientist AI partner.

They have analyzed zone: {zone}. Here's their summary:
\"\"\"{summary}\"\"\"

Respond with:
1. Encouragement or fun comment,
2. A quirky ecological insight or analogy,
3. Optionally suggest a small improvement or next step.
"""
