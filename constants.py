# constants.py

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Default model name to use (customize as needed)
MODEL_NAME = "llama2:7b"

# Request timeout in seconds
REQUEST_TIMEOUT = 300

# Maximum number of rows to include in prompt (avoid overly large prompts)
MAX_CSV_ROWS_IN_PROMPT = 10