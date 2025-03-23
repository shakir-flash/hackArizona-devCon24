import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Import constants from constants.py
from constants import OLLAMA_URL, MODEL_NAME, REQUEST_TIMEOUT

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def guess_zone_from_filename(filename: str) -> str:
    """
    Naively guess the zone based on the filename.
    """
    fname = filename.lower()
    if "ocean" in fname:
        return "Ocean"
    elif "desert" in fname:
        return "Desert"
    elif "rainforest" in fname or "rf" in fname:
        return "Rainforest"
    elif "leo" in fname:
        return "LEO"
    else:
        return None

def build_prompt(dataset_name: str, sample_data: pd.DataFrame) -> str:
    """
    Build the prompt to send to Gemma3 using a preview of the combined data,
    preceded by a base context describing Biosphere 2.
    """
    base_context = (
        "Biosphere 2 is a 3.14-acre research and education campus near Oracle, Arizona. "
        "It has been owned by the University of Arizona (UArizona) since 2011, though its history dates back to the 1800s when the land was part of the Samaniego CDO Ranch. "
        "In the 1980s, Space Biospheres Ventures constructed the iconic glass-enclosed facility to study self-sustaining space-colony technology. Notably, two missions from 1991 to 1994 saw researchers sealed inside for months, testing the viability of closed ecological systems.\n\n"
        "Today, Biosphere 2 serves as a world-class hub for climate and sustainability research. It features multiple biomes under one roof—ocean, mangrove wetlands, tropical rainforest, savanna grassland, and fog desert—supported by a sophisticated “technosphere” (basement) that controls temperature, humidity, and airflow. The campus also includes administrative offices, classrooms, labs, and a conference center, altogether hosting more than three million visitors since its inception (including over half a million K–12 students).\n\n"
        "Under UArizona’s stewardship, Biosphere 2 advances interdisciplinary experiments, such as the Landscape Evolution Observatory, while fostering innovative solutions in areas ranging from climate change to ecosystem resilience and sustainable development. Strategically, it aims to become a global center for “resilience solutions” by 2030, addressing grand challenges in biodiversity and sustainability and contributing research, education, and insights both on Earth and in potential space habitats.\n\n"
        "Rainforest Biome Zone:\n"
        "Biosphere 2’s 20,000 sq. ft. tropical rainforest is modeled on the Amazon Basin. Initially populated with over 400 species (2,800 individual plants), it now contains around 100 plant species and various insects. Research focuses on plant-atmosphere interactions—how rainforest plants exchange gases and adapt to water stress, which informs climate change predictions. The biome is divided into multiple microhabitats, including lowland rainforest, terraces (coffee, papaya), a ginger belt, bamboo belt (to block ocean salt), varzea (seasonal floodplain), and tepui (cloud forest).\n\n"
        "Ocean Biome Zone:\n"
        "Originally designed to replicate a Caribbean reef, the 2.6 million-liter marine mesocosm is the largest enclosed ocean research system in the world. It allows precise control over temperature, chemistry, and light to study coral reef dynamics and resilience. Past experiments by Columbia University showed significant coral calcification decline under elevated CO₂ levels. After degradation in the early 2000s, the system is now being revitalized to support diverse reef zones (fore-reef, crest, lagoon) and is home to fish, hermit crabs, urchins, snails, and anemones.\n\n"
        "Landscape Evolution Observatory (LEO) Zone:\n"
        "LEO is a large-scale experiment simulating how landscapes evolve under changing climate conditions. It bridges the gap between lab and field studies by integrating sensors and models to track interactions among water, soil, microbes, and plants. This macrocosm enables scientists to predict how Earth’s critical zone—where water, air, rock, and life interact—responds to environmental stress over time.\n\n"
        "Desert Biome Zone:\n"
        "The coastal fog desert mimics arid scrubland with seasonal rains and summer droughts. Originally dense due to early overwatering, it has been restructured to support more drought-tolerant plants while discouraging invasive grasses. Features include synthetic soils, mini-rhizotron tubes for root observation, a salt-accumulating playa dominated by saltbushes, and a tinaja (rock basin) for freshwater species. Biodiversity has declined, partly due to management shifts and loss of pollinators.\n\n"
    )
    sample_preview = sample_data.head(5).to_csv(index=False)
    prompt = (
        f"{base_context}\n\n"
        f"You are an intelligent environmental agent operating in Biosphere 2.\n"
        f"You are now inside the zone: **{dataset_name}**.\n\n"
        f"Here is a preview of current sensor readings in this zone:\n"
        f"{sample_preview}\n\n"
        f"Based on the data above, what balancing actions or insights should you generate "
        f"to maintain optimal conditions in this zone, if any?\n\n"
        f"List actionable recommendations as if you are managing this system."
    )
    return prompt

def query_llm(prompt: str) -> str:
    """
    Query the LLM via Ollama with the given prompt.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT
        )
        return response.json().get("response", "⚠️ No response from LLM.")
    except Exception as e:
        return f"Error querying LLM: {e}"

def determine_severity(llm_response: str) -> str:
    """
    Naively determine severity based on keywords in the LLM response.
    Returns "red" if severe, "yellow" if moderate, else "normal".
    """
    lower_resp = llm_response.lower()
    if "urgent" in lower_resp or "critical" in lower_resp:
        return "severe"
    elif "issue" in lower_resp or "problem" in lower_resp or "concern" in lower_resp:
        return "moderate"
    else:
        return "normal"

# -------------------------
# BASE COLORS & UTILITY FUNCTIONS
# -------------------------
BASE_COLORS = {
    "Ocean": "#66B2FF",       # Tropical Blue
    "Desert": "#E4C580",      # Sandy Color
    "Rainforest": "#006400",  # Lush Dark Green
    "LEO": "#708090"          # Slate Gray for LEO
}

def get_zone_color(zone: str, severity: str) -> str:
    if severity == "red":
        return "#FF6347"  # Tomato Red
    elif severity == "yellow":
        return "#FFD700"  # Gold
    else:
        return BASE_COLORS.get(zone, "#BBBBBB")

# -------------------------
# SESSION STATE INITIALIZATION
# -------------------------
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []  # List of uploaded files
if "zone_files" not in st.session_state:
    st.session_state.zone_files = {}      # Map: zone -> list of files
if "analysis_log" not in st.session_state:
    st.session_state.analysis_log = {}    # Map: zone -> analysis result
if "zone_status" not in st.session_state:
    st.session_state.zone_status = {
        "Ocean": "normal",
        "Desert": "normal",
        "Rainforest": "normal",
        "LEO": "normal"
    }
if "selected_zone" not in st.session_state:
    st.session_state.selected_zone = None
if "selected_files" not in st.session_state:
    st.session_state.selected_files = []
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # List of messages (each is a dict with 'role' and 'content')

# -------------------------
# STREAMLIT UI SETUP
# -------------------------
st.set_page_config(page_title="Biosphere 2 - Zone Dashboard", layout="wide")
st.title("Biosphere 2: Zone Dashboard")

# Sidebar: File Uploader for Multiple Files
uploaded_files = st.sidebar.file_uploader("Upload sensor data file(s)", type=["csv", "xlsx"], accept_multiple_files=True)
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    # Group files by zone using the filename heuristic
    zone_files = {}
    for file in st.session_state.uploaded_files:
        zone = guess_zone_from_filename(file.name)
        if zone:
            zone_files.setdefault(zone, []).append(file)
    st.session_state.zone_files = zone_files

# Sidebar: Zone Selector Dropdown (only zones with uploaded files)
if st.session_state.zone_files:
    available_zones = list(st.session_state.zone_files.keys())
    selected_zone = st.sidebar.selectbox("Select a zone to analyze", available_zones)
    # When a new zone is selected, reset grid status and conversation history
    st.session_state.zone_status = { "Ocean": "normal", "Desert": "normal", "Rainforest": "normal", "LEO": "normal" }
    st.session_state.conversation = []
    st.session_state.selected_zone = selected_zone

    # Sidebar: Multiselect for files within the selected zone
    zone_file_names = [f.name for f in st.session_state.zone_files[selected_zone]]
    selected_files = st.sidebar.multiselect("Select file(s) for analysis", zone_file_names, default=zone_file_names)
    st.session_state.selected_files = selected_files

    # Analyze Button: Process the selected files for the zone
    if st.sidebar.button("Analyze Selected Files"):
        files_to_analyze = [f for f in st.session_state.zone_files[selected_zone] if f.name in st.session_state.selected_files]
        if files_to_analyze:
            # Reset grid status for all zones before analysis
            st.session_state.zone_status = { "Ocean": "normal", "Desert": "normal", "Rainforest": "normal", "LEO": "normal" }
            dfs = []
            for file in files_to_analyze:
                try:
                    if file.name.endswith(".csv"):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    dfs.append(df)
                except Exception as e:
                    st.sidebar.error(f"Error reading {file.name}: {e}")
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                with st.spinner("Analyzing combined sensor data..."):
                    prompt_text = build_prompt(selected_zone, combined_df)
                    llm_result = query_llm(prompt_text)
                    new_severity = determine_severity(llm_result)
                    # If there's an initial analysis, compare severity; if unchanged, keep initial severity.
                    if selected_zone in st.session_state.analysis_log:
                        initial_severity = st.session_state.analysis_log[selected_zone]["severity"]
                        if new_severity == initial_severity:
                            severity = initial_severity
                        else:
                            severity = new_severity
                    else:
                        severity = new_severity
                    st.session_state.zone_status[selected_zone] = severity
                    st.session_state.analysis_log[selected_zone] = {
                        "llm_output": llm_result,
                        "severity": severity,
                        "files": st.session_state.selected_files
                    }
                st.sidebar.success("Analysis complete for selected files!")
        else:
            st.sidebar.warning("No files selected for analysis.")

# -------------------------
# MAIN DASHBOARD: 1x4 Grid and Analysis Log
# -------------------------
st.markdown("## Zone Status")

# 1x4 Grid Display
col1, col2, col3, col4 = st.columns(4)
def render_zone_box(col, zone_name):
    severity = st.session_state.zone_status.get(zone_name, "normal")
    box_color = get_zone_color(zone_name, severity)
    zone_html = f"""
    <div style="background-color:{box_color}; padding:30px; border-radius:10px; text-align:center">
        <h3 style="color:white;">{zone_name}</h3>
        <p style="color:white;">Status: {severity.upper()}</p>
    </div>
    """
    col.markdown(zone_html, unsafe_allow_html=True)

render_zone_box(col1, "Ocean")
render_zone_box(col2, "Desert")
render_zone_box(col3, "Rainforest")
render_zone_box(col4, "LEO")

# Display the Analysis Log for the selected zone below the grid
if st.session_state.selected_zone:
    st.markdown("---")
    st.subheader(f"Analysis for Zone: {st.session_state.selected_zone}")
    analysis = st.session_state.analysis_log.get(st.session_state.selected_zone)
    if analysis:
        st.text_area("LLM Output", value=analysis["llm_output"], height=250)
        st.markdown(f"**Severity:** {analysis['severity'].upper()}")
        severity = analysis['severity'].upper()
        st.markdown(f"**Files Used for Analysis:** {', '.join(analysis['files'])}")
    else:
        st.info("No analysis available for the selected zone.")

# -------------------------
# CONVERSATION SECTION
# -------------------------
if st.session_state.selected_zone and st.session_state.selected_zone in st.session_state.analysis_log:
    st.markdown("## Conversation on Analysis")
    conv_placeholder = st.empty()
    
    def display_conversation():
        conversation_content = ""
        if st.session_state.conversation:
            for msg in st.session_state.conversation:
                if msg["role"] == "User":
                    conversation_content += f"**User:** {msg['content']}\n\n"
                else:
                    conversation_content += f"**Assistant:** {msg['content']}\n\n"
        return conversation_content
    
    conv_history = display_conversation()
    conv_placeholder.markdown(conv_history)
    
    # Standard conversation input
    new_message = st.text_input("Your message:", key="conversation_input")
    
    col_send, col_update = st.columns(2)
    if col_send.button("Send Message"):
        if new_message:
            st.session_state.conversation.append({"role": "User", "content": new_message})
            if len(st.session_state.conversation) > 5:
                st.session_state.conversation = st.session_state.conversation[-5:]
            analysis_context = st.session_state.analysis_log[st.session_state.selected_zone]["llm_output"]
            conv_prompt = f"Analysis Context: {analysis_context}\n"
            for msg in st.session_state.conversation:
                conv_prompt += f"{msg['role']}: {msg['content']}\n"
            conv_prompt += "Assistant: "
            with st.spinner("Generating assistant reply..."):
                assistant_reply = query_llm(conv_prompt)
            st.session_state.conversation.append({"role": "Assistant", "content": assistant_reply})
            if len(st.session_state.conversation) > 5:
                st.session_state.conversation = st.session_state.conversation[-5:]
            conv_history = display_conversation()
            conv_placeholder.markdown(conv_history)
    
    # New sensor data update via file upload (entire file appended)
    new_data_file = st.file_uploader("Upload new sensor data to update insight", type=["csv", "xlsx"], key="new_sensor_data_upload")
    if new_data_file:
        try:
            if new_data_file.name.endswith(".csv"):
                new_data_df = pd.read_csv(new_data_file)
            else:
                new_data_df = pd.read_excel(new_data_file)
            # Use the entire file instead of just the first 5 rows
            additional_data_full = new_data_df.to_csv(index=False)
        except Exception as e:
            st.error(f"Error reading new sensor data: {e}")
            additional_data_full = "Error reading file."
        
        if col_update.button("Update Insight with New Data"):
            update_message = f"New sensor data update:\n{additional_data_full}"
            st.session_state.conversation.append({"role": "User", "content": update_message})
            if len(st.session_state.conversation) > 5:
                st.session_state.conversation = st.session_state.conversation[-5:]
            analysis_context = st.session_state.analysis_log[st.session_state.selected_zone]["llm_output"]
            conv_prompt = f"Analysis Context: {analysis_context}\n"
            for msg in st.session_state.conversation:
                conv_prompt += f"{msg['role']}: {msg['content']}\n"
            conv_prompt += "Assistant: "
            with st.spinner("Generating updated insight..."):
                assistant_reply = query_llm(conv_prompt)
            st.session_state.conversation.append({"role": "Assistant", "content": assistant_reply})
            if len(st.session_state.conversation) > 5:
                st.session_state.conversation = st.session_state.conversation[-5:]
            conv_history = display_conversation()
            conv_placeholder.markdown(conv_history)