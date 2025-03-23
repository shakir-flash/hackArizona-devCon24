# digital_twin_simulator.py

import streamlit as st
import requests
import pandas as pd
from constants import DATA_DIR, OLLAMA_URL, MODEL_NAME
from dataset_loader import load_zone_datasets
from prompt_engine import build_prompt, build_small_talk_prompt
from ml_utils_simple import train_and_predict

st.set_page_config("B2Twin - Digital Twin Navigator", layout="wide")
st.title("🌿 B2Twin - AI-Powered Ecosystem Insight Agent")

# Header-level link to second Streamlit app
st.markdown(
    """
    <div style='margin-top:-15px; font-size:16px;'>
        🔗 <a href="https://devcon4-team25-1.streamlit.app/" target="_blank" style="text-decoration:none; color:#0072C6; font-weight:bold;">
        👉 Launch Conversational AI - Sequential Agent
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# Load datasets
if "datasets" not in st.session_state:
    st.session_state.datasets = load_zone_datasets(DATA_DIR)
    st.session_state.zone_list = list(st.session_state.datasets.keys())
    st.session_state.zone_index = 0
    st.session_state.logs = []

zones = st.session_state.datasets
zone_names = st.session_state.zone_list
index = st.session_state.zone_index
current_zone = zone_names[index]
df = zones[current_zone]

# CSV Uploader
st.markdown("---")
st.subheader("📤 Upload Your Own Dataset (CSV)")
uploaded_file = st.file_uploader("Upload a .csv file", type=["csv"])
if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    uploaded_df.columns = [col.strip().lower().replace(" ", "_") for col in uploaded_df.columns]
    st.session_state.datasets["Uploaded CSV"] = uploaded_df
    if "Uploaded CSV" not in st.session_state.zone_list:
        st.session_state.zone_list.append("Uploaded CSV")
    st.success("✅ Uploaded CSV added as new zone")

# Current Zone Viewer
st.markdown("---")
st.subheader(f"📍 Current Zone: {current_zone}")
st.dataframe(df.head(10))

col1, col2 = st.columns(2)
with col1:
    if st.button("⬅️ Previous Zone"):
        st.session_state.zone_index = (index - 1) % len(zone_names)
with col2:
    if st.button("➡️ Next Zone"):
        st.session_state.zone_index = (index + 1) % len(zone_names)

# ML Training
st.markdown("---")
st.subheader("📈 In-Memory ML Training & Prediction")
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if numeric_cols:
    target_col = st.selectbox("Select target variable", numeric_cols)
    if st.button("Train + Predict"):
        model, preds = train_and_predict(df, target_col)
        if preds is not None:
            st.line_chart(preds[:100])
            st.success("✅ Predictions ready")
        else:
            st.warning("⚠️ Model failed to train/predict")
else:
    st.warning("No numeric columns available")

# Main LLM Analysis
st.markdown("---")
st.subheader("🤖 LLM Scientific Insight (Main Agent)")
if st.button("Ask LLM for Scientific Analysis"):
    prompt = build_prompt(current_zone, df)
    try:
        res = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False})
        json_resp = res.json()
        response_text = json_resp.get("response") or json_resp.get("output") or str(json_resp)
        st.text_area("LLM Response", response_text, height=250)
        st.session_state.logs.append(f"{current_zone}: {response_text}")
        st.session_state.last_llm_response = response_text
    except Exception as e:
        st.error(f"❌ LLM error: {str(e)}")

# AI Agent Log
st.markdown("---")
st.subheader("📘 AI Agent Memory Log")
st.text_area("Memory Log", "\n".join(st.session_state.logs), height=300)

# Ecosystem Health Tracker
st.markdown("---")
st.subheader("🏁 Mission Ecosystem Health Tracker")
health_count = 0
total_zones = len(zone_names)

for z in zone_names:
    df_z = zones[z]
    rh_col = [c for c in df_z.columns if "rh" in c.lower()]
    temp_col = [c for c in df_z.columns if "temp" in c.lower()]
    co2_col = [c for c in df_z.columns if "co2" in c.lower()]
    status = "✅ Healthy"
    color = "green"
    try:
        if temp_col and df_z[temp_col[0]].mean() > 35:
            status = "🔥 Hot"
            color = "red"
        elif co2_col and df_z[co2_col[0]].mean() > 700:
            status = "☣ High CO2"
            color = "orange"
        elif rh_col and df_z[rh_col[0]].mean() < 30:
            status = "💨 Low RH"
            color = "blue"
        else:
            health_count += 1
    except:
        status = "❓ Check Data"
        color = "gray"
    st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:5px;margin:5px'>{z} ➜ {status}</div>", unsafe_allow_html=True)

progress = health_count / total_zones
st.progress(progress)
st.metric("Zones Stable", f"{health_count}/{total_zones}")

# Inter-AI Communication Demo
st.markdown("---")
st.subheader("🔗 Inter-AI Communication Protocol Demo")
demo_port = st.text_input("Enter Recipient AI Port", value="5001")
demo_message = f"🌱 AI Agent update from zone: {current_zone}.\nSample sensor snapshot:\n{df.describe().to_string()}"

if st.button("🚀 Send Demo Message to Other AI Agent"):
    try:
        demo_payload = {"from_agent": "B2Twin-AI-Agent-Demo", "message": demo_message}
        demo_url = f"http://localhost:{demo_port}/receive-message"
        response = requests.post(demo_url, json=demo_payload)
        response_json = response.json()
        st.success(f"✅ Inter-AI Response: {response_json.get('response')}")
    except Exception as e:
        st.error(f"❌ Failed to communicate with other AI: {e}")

# Small Talk Assistant AI
st.markdown("---")
st.subheader("🧠 Assistant AI Small Talk Collaboration")
if "last_llm_response" not in st.session_state:
    st.session_state.last_llm_response = "No previous response from main agent."
st.text_area("Main Agent Last Summary", st.session_state.last_llm_response, height=200)

if st.button("🤖 Talk to Assistant AI Agent"):
    try:
        small_talk_prompt = build_small_talk_prompt(current_zone, st.session_state.last_llm_response)
        res = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": small_talk_prompt, "stream": False})
        assistant_reply = res.json().get("response") or res.json().get("output") or "No response"
        st.text_area("Assistant AI Reply", assistant_reply, height=200)
        st.session_state.logs.append(f"AssistantAI to {current_zone}: {assistant_reply}")
    except Exception as e:
        st.error(f"❌ Assistant AI error: {str(e)}")
