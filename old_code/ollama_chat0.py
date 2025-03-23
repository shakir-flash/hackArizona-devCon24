import streamlit as st
import requests

# Import your constants
import constants

st.set_page_config(page_title="Local Chatbot (Ollama + gemma3)", layout="centered")

# 1. Load external CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🤖 Local Chatbot (Ollama + gemma3)")

# 2. Inject JavaScript to scroll to the latest user message (with id "latest-user")
scroll_js = """
<script>
function scrollToLatest() {
    var latest = window.parent.document.getElementById("latest-user");
    if (latest) {
        latest.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}
</script>
"""
st.markdown(scroll_js, unsafe_allow_html=True)

# 3. Initialize chat history in session state
#    Each message is a dict with keys: "role", "content", "file_name", "file_type"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def send_prompt_to_llm(prompt_text: str) -> str:
    """
    Send the given prompt text to the local LLM via Ollama
    and return the AI response or an error message.
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
        return "Error: Request to local LLM timed out."
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to local LLM. Is Ollama running?"
    except Exception as e:
        return f"Error: {str(e)}"

def display_chat_history():
    """
    Renders the entire chat history in a chat-like format.
    Right-aligned user messages (with file info if present)
    and left-aligned AI messages.

    The latest user message gets an id ("latest-user") so
    we can auto-scroll there.
    """
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Find index of the last user message
    last_user_index = None
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            last_user_index = i

    # Display messages
    for i, msg in enumerate(st.session_state.chat_history):
        role = msg["role"]
        content = msg["content"]
        file_name = msg["file_name"]
        file_type = msg.get("file_type", None)

        # If this is the last user message, add id="latest-user"
        extra_attr = ""
        if role == "user" and i == last_user_index:
            extra_attr = "id='latest-user'"

        if role == "user":
            # If a file is attached, display the file name and type
            if file_name:
                # Show something like "File: Desert_CO2_FEB_2025.csv (text/csv)"
                # Or just the extension if you prefer
                if file_type:
                    st.markdown(
                        f"<div class='chat-message user-message' {extra_attr}>"
                        f"<div class='filename-display'>File: {file_name} ({file_type})</div>"
                        f"{content}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='chat-message user-message' {extra_attr}>"
                        f"<div class='filename-display'>File: {file_name}</div>"
                        f"{content}</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    f"<div class='chat-message user-message' {extra_attr}>{content}</div>",
                    unsafe_allow_html=True
                )
        else:
            # AI message (left-aligned)
            st.markdown(
                f"<div class='chat-message ai-message'>{content}</div>",
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

# 4. Create a single form with file uploader + user prompt
with st.form(key="chat_form", clear_on_submit=True):
    uploaded_file = st.file_uploader(
        "Attach a file (optional)",
        type=["csv", "txt", "docx", "pdf"], 
        label_visibility="visible"
    )
    user_input = st.text_input(
        "Your prompt:",
        placeholder="Ask something about your file or Biosphere 2..."
    )
    submit_button = st.form_submit_button("Send")

if submit_button:
    # Store file name + file type, but DO NOT parse or show contents
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_type = uploaded_file.type  # e.g. "text/csv" or "application/pdf"
    else:
        file_name = None
        file_type = None

    # Add the user's message to the chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "file_name": file_name,
        "file_type": file_type
    })

    # We do NOT attach any file snippet to the prompt here — we just send user_input
    with st.spinner("Thinking..."):
        ai_response = send_prompt_to_llm(user_input)

    # Add the AI response to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": ai_response,
        "file_name": None,
        "file_type": None
    })

    # Auto-scroll to the latest user message
    st.markdown("<script>scrollToLatest();</script>", unsafe_allow_html=True)

st.markdown("---")
display_chat_history()
st.markdown("<script>scrollToLatest();</script>", unsafe_allow_html=True)