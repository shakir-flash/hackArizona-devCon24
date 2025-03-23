# digital_twin_simulator.py

import streamlit as st
import requests
import numpy as np
from constants import OLLAMA_URL, MODEL_NAME, REQUEST_TIMEOUT, DATA_DIR
from dataset_loader import load_zone_datasets
from prompt_engine import build_zone_prompt
from zone_navigator import next_zone, previous_zone
from ml_trainer import train_zone_model, predict_zone_model
from online_trainer_river import initialize_online_model, update_online_model, predict_online_model

st.set_page_config(page_title="B2Twin Digital Twin", layout="wide")
st.title("🌿 B2Twin - AI Digital Twin Navigator")

if "datasets_loaded" not in st.session_state:
    st.session_state.zones = load_zone_datasets(DATA_DIR)
    st.session_state.zone_keys = list(st.session_state.zones.keys())
    st.session_state.zone_index = 0
    st.session_state.agent_log = []
    st.session_state.zone_goals = {}
    st.session_state.zone_status = {}
    st.session_state.datasets_loaded = True

zones = st.session_state.zones
if not zones:
    st.warning("⚠️ No datasets found.")
    st.stop()

zone_keys = st.session_state.zone_keys
zone_idx = st.session_state.zone_index
current_zone = zone_keys[zone_idx]
df = zones[current_zone]

st.subheader(f"📍 Current Zone: {current_zone}")
st.dataframe(df.head(10))

col1, col2 = st.columns(2)
with col1:
    if st.button("⬅️ Previous Zone"):
        st.session_state.zone_index, current_zone = previous_zone(zone_idx, zone_keys)
with col2:
    if st.button("➡️ Next Zone"):
        st.session_state.zone_index, current_zone = next_zone(zone_idx, zone_keys)

# Goals
st.subheader("🎯 Zone Goals")
def check_goals(df, zone):
    goals = {"temp": (18, 30), "co2": (300, 600), "rh": (40, 80), "par": (200, 1000)}
    status = True
    lines = []
    for col in df.columns:
        for key, (low, high) in goals.items():
            if key in col:
                val = df[col].mean(skipna=True)
                if low <= val <= high:
                    lines.append(f"✅ {col}: {val:.2f} OK")
                else:
                    lines.append(f"⚠️ {col}: {val:.2f} out of range")
                    status = False
    return lines, status

goal_lines, zone_ok = check_goals(df, current_zone)
st.session_state.zone_status[current_zone] = zone_ok
for line in goal_lines:
    st.markdown(line)

# Batch Training
st.subheader("🧠 ML Batch Training")
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
target_col = st.selectbox("Target Column", numeric_cols)
if st.button("📈 Train Batch Model"):
    st.success(train_zone_model(df, current_zone, target_col))
if st.button("🔮 Predict (Batch)"):
    preds = predict_zone_model(df, current_zone, target_col)
    if preds is not None:
        st.line_chart(preds[:100])

# Online Learning
st.subheader("🚀 Online Learning")
if st.button("Init Online Model"):
    initialize_online_model(current_zone)
    st.success("Initialized Online Model")
if st.button("Update Online with 1st row"):
    row = df.select_dtypes(include=["float64", "int64"]).dropna().iloc[0]
    update_online_model(current_zone, row.drop(target_col).to_dict(), row[target_col])
    st.success("Model Updated")
if st.button("Predict Online"):
    row = df.select_dtypes(include=["float64", "int64"]).dropna().iloc[1]
    pred = predict_online_model(current_zone, row.drop(target_col).to_dict())
    st.metric("Online Prediction", f"{pred:.2f}")

# LLM + Auto-Actions
st.subheader("🤖 LLM Suggestions & Auto-Actions")
prompt = build_zone_prompt(current_zone, df)
if st.button("Ask LLM & Simulate Actions"):
    response = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt}, timeout=REQUEST_TIMEOUT)
    text = response.json().get("response", "No LLM response")
    st.text_area("LLM Output", text)
    st.session_state.agent_log.append(f"[{current_zone}] {text}")
    if "increase rh" in text.lower():
        for col in df.columns:
            if "rh" in col: df[col] += 10
    if "decrease co2" in text.lower():
        for col in df.columns:
            if "co2" in col: df[col] -= 50
    if "adjust temp" in text.lower():
        for col in df.columns:
            if "temp" in col: df[col] -= 2

# Logs
st.subheader("📜 Agent Logs")
st.text_area("Agent Log", "\n".join(st.session_state.agent_log), height=300)

# Mission Tracker
st.subheader("🏁 Mission Status")
total = len(st.session_state.zone_status)
success = sum(1 for v in st.session_state.zone_status.values() if v)
if success == total:
    st.balloons()
    st.success("🎉 MISSION COMPLETE!")
else:
    st.info(f"{success}/{total} zones healthy")
