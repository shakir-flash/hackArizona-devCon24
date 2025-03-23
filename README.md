# 🏆 HackArizona-DevCon24: B2Twin AI Project

Below is a personalized summary of everything ew accomplished within this repository. It's a behind-the-scenes look at all the core tasks we tackled.

---

## **1. Data Preparation**

- **Cleaned and standardized 24 Biosphere 2 sensor CSVs**  
  Handled missing values, standardized temperature units, and ensured consistent column naming conventions.
- **Implemented a lightweight data-loading pipeline**  
  Created `dataset_loader.py` to load each zone’s CSV into memory with minimal overhead.

---

## **2. Local AI Model & LLM Integration**

- **In-memory ML training** (`ml_utils_simple.py`)  
  – Used a simple linear regression approach to quickly demo predictions on any numeric column.  
  – Avoided complex joblib saving/loading to minimize deployment errors.

- **Local LLM usage** (`prompt_engine.py` + `digital_twin_simulator.py`)  
  – Deployed a local Gemma 3-based Large Language Model via Ollama.  
  – Engineered prompts for both main scientific analysis and a “small talk” assistant agent.

---

## **3. Streamlit UI & Ecosystem Simulation** (`digital_twin_simulator.py`)

- **Developed a Streamlit-based interface** to:
  1. Upload brand-new CSVs on the fly.  
  2. Navigate zones with Previous/Next buttons.  
  3. Train/predict with a simple ML model.  
  4. Query the local LLM for scientific insights.  
  5. Track a “mission health” metric across all zones.

- **Multi-agent demonstration**  
  – Created an assistant LLM prompt to add more personality and collaboration in the final hackathon demo.

---

## **4. Inter-AI Communication Protocol** (`ai_comm_server.py`)

- **Built a Flask server** to accept JSON messages from other AI agents.  
- **Added a “Send Message” block** in the Streamlit UI for port-based communication.  
- Demonstrated multi-agent synergy and potential for scaling to a bigger AI network.

---

## **5. Old Code & Prototype Exploration** (`old_code/`)

- **Maintained an archive** of older scripts:
  - Early advanced data cleaning
  - Online training with River
  - Early LLM chat prototypes
- **Ensured no confusion** by keeping the final approach minimal: no external joblib or advanced code complexities.

---

## **6. Deployment & Testing**

- **Streamlined the environment** with a single `requirements.txt`, removing incompatible dependencies.  
- **Tested locally** with `streamlit run` for the main UI and `python ai_comm_server.py` for port-based comms.  
- Overcame repeated file path issues by simplifying model usage (in-memory only) to ensure stable, error-free demos.

---

## **What This Achieves**

- A robust yet lightweight digital twin demonstration, focusing on **actionable insights** from Biosphere 2 data.  
- Full local LLM integration, minimal overhead, multi-zone approach, plus a creative assistant AI for “small talk.”  
- Clear path to expand or integrate with real-time data feeds and bigger frameworks if needed.

---

**Thank you for exploring our HackArizona DevCon24 B2Twin Project!**  
Feel free to build on this foundation, adapt it to new data, or extend the multi-agent capabilities for deeper ecological simulations. Enjoy! 
