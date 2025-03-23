# digital_twin_simulator.py

import streamlit as st
from constants import OLLAMA_URL, MODEL_NAME, REQUEST_TIMEOUT
from dataset_loader import load_zone_datasets
from prompt_engine import build_zone_prompt
from zone_navigator import next_zone, previous_zone
import requests

st.set_page_config(page_title="Biosphere2 Digital Twin Navigator", layout="wide")
st.title("🌿 B2Twin - AI Agent Navigator for Biosphere 2")

# Step 1: Load datasets
data_folder = st.sidebar.text_input("📁 Data Folder Path:", value="data/clean_data")
if st.sidebar.button("🔄 Load Zones"):
    st.session_state.zones = load_zone_datasets(data_folder)
    st.session_state.zone_keys = list(st.session_state.zones.keys())
    st.session_state.zone_index = 0
    st.session_state.agent_log = []
    st.success("✅ Datasets loaded as independent zones!")

# Ensure datasets loaded
if "zones" in st.session_state and st.session_state.zones:
    zones = st.session_state.zones
    zone_keys = st.session_state.zone_keys
    zone_idx = st.session_state.zone_index
    current_zone = zone_keys[zone_idx]
    df = zones[current_zone]

    # Display current zone info
    st.subheader(f"📍 Current Zone: {current_zone}")
    st.dataframe(df.head(10))

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Previous Zone"):
            st.session_state.zone_index, current_zone = previous_zone(zone_idx, zones)
    with col2:
        if st.button("➡️ Next Zone"):
            st.session_state.zone_index, current_zone = next_zone(zone_idx, zones)

    # Build prompt
    prompt = build_zone_prompt(current_zone, df)

    # Ask LLM
    if st.button("🧠 Ask LLM to Analyze This Zone"):
        try:
            with st.spinner("Consulting LLM..."):
                response = requests.post(
                    OLLAMA_URL,
                    json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
                    timeout=REQUEST_TIMEOUT
                )
                result = response.json().get("response", "⚠️ No response from LLM.")
                st.text_area("🔍 LLM Recommendations", value=result, height=250)
                st.session_state.agent_log.append(f"[{current_zone}] {result}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    # Agent Log
    st.subheader("📝 Agent Log")
    log_content = "\n".join(st.session_state.agent_log)
    st.text_area("Agent Action History", value=log_content, height=300)
else:
    st.warning("Upload your datasets and click 'Load Zones' first.")
