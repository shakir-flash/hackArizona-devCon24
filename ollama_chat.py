# ollama_chat.py

import streamlit as st
import requests
from advanced_data_cleaning import load_and_advanced_clean_all_csvs
from data_utils import extended_summarize_df
from prompt_utils import (
    build_descriptive_prompt,
    build_analytical_prompt,
    build_predictive_prompt,
    build_hypothesis_generation_prompt
)
from custom_inputs import (
    get_dataset_custom_context,
    customize_descriptive_input,
    customize_analytical_input,
    customize_predictive_input,
    customize_hypothesis_generation_input
)
import constants  # Import configuration constants

# Set page configuration for Streamlit
st.set_page_config(page_title="Local Chatbot (Gemma3 4b)", layout="wide")

# Load custom CSS for UI (if available)
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ========= SIDEBAR: Data Loading & Analysis Method ========= #
st.sidebar.title("Data & Analysis Setup")
data_folder = st.sidebar.text_input("Data Folder Path:", value="data")
load_button = st.sidebar.button("Load & Preprocess Data")

# Initialize session state for CSVs and summaries
if "csv_files" not in st.session_state:
    st.session_state.csv_files = {}
if "summaries" not in st.session_state:
    st.session_state.summaries = {}

if load_button:
    with st.spinner("Loading and cleaning data..."):
        cleaned_files = load_and_advanced_clean_all_csvs(data_folder)
        st.session_state.csv_files = cleaned_files
        # Generate extended summaries for each cleaned DataFrame
        summaries = {fname: extended_summarize_df(df, fname) for fname, df in cleaned_files.items()}
        st.session_state.summaries = summaries
    st.success("Data loaded, cleaned, and summarized successfully!")

# ========= MAIN APP: Chat Interface ========= #
st.markdown("<h1 style='text-align: center;'>🤖 Local Chatbot (Gemma3 4b)</h1>", unsafe_allow_html=True)
st.write("""
**Instructions:**
1. Load & Preprocess Data from the sidebar.
2. Select a data file.
3. Choose a prompt type for LLM Analysis.
4. Click "Generate LLM Prompt & Get Analysis" to send a customized prompt to the LLM.
""")

if st.session_state.summaries:
    all_files = list(st.session_state.summaries.keys())
    selected_file = st.selectbox("Select a data file for analysis:", all_files)
    summary_text = st.session_state.summaries[selected_file]
    df_selected = st.session_state.csv_files[selected_file]
    custom_context = get_dataset_custom_context(df_selected, selected_file)
    
    prompt_types = ["Descriptive", "Analytical", "Predictive", "Hypothesis Generation"]
    prompt_type = st.selectbox("Select Prompt Type:", prompt_types)
    
    def build_prompt():
        if prompt_type == "Descriptive":
            return customize_descriptive_input(summary_text, custom_context)
        elif prompt_type == "Analytical":
            return customize_analytical_input(summary_text, custom_context)
        elif prompt_type == "Predictive":
            return customize_predictive_input(summary_text, custom_context)
        elif prompt_type == "Hypothesis Generation":
            return customize_hypothesis_generation_input(summary_text, custom_context)
        else:
            return "No valid prompt selected."
    
    def send_prompt_to_llm(prompt_text: str) -> str:
        """
        Sends the prompt to the local LLM service and returns the response.
        """
        try:
            response = requests.post(
                constants.OLLAMA_URL,
                json={
                    "model": constants.MODEL_NAME,
                    "prompt": prompt_text,
                    "stream": False
                },
                timeout=constants.REQUEST_TIMEOUT
            )
            if response.status_code != 200:
                return f"Error: Server returned status code {response.status_code}"
            json_data = response.json()
            if "response" not in json_data:
                return "Error: 'response' field not found in JSON"
            if not json_data["response"]:
                return "Error: No response from AI"
            return json_data["response"]
        except requests.exceptions.Timeout:
            return "Error: Request timed out."
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to the LLM service. Is it running?"
        except Exception as e:
            return f"Error: {str(e)}"
    
    if st.button("Generate LLM Prompt & Get Analysis"):
        final_prompt = build_prompt()
        with st.spinner("Processing LLM prompt..."):
            ai_response = send_prompt_to_llm(final_prompt)
        st.subheader("LLM Prompt")
        st.text_area("Prompt Sent to LLM:", final_prompt, height=150)
        st.subheader("LLM Response")
        st.text_area("LLM Response:", ai_response, height=200)
else:
    st.warning("No data loaded yet. Please load & preprocess the CSV files from the sidebar.")
