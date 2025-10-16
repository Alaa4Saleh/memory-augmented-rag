# config.py
import os
from dotenv import load_dotenv
from pathlib import Path # Import Path from pathlib

# --- Absolute path loading for .env ---
# Get the directory where config.py itself resides
config_dir = Path(__file__).resolve().parent
# Construct the absolute path to the .env file
dotenv_path = config_dir / '.env'

# Load environment variables using the absolute path
load_dotenv(dotenv_path=dotenv_path)

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Using os.getenv with a default fallback is good practice here too
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro-preview-03-25")

# Ensure API key is loaded
if GOOGLE_API_KEY is None:
    raise ValueError(
        f"GOOGLE_API_KEY not found in environment variables. "
        f"Expected .env file at: {dotenv_path}"
    )

# Dataset Configuration
NUM_CONVERSATIONS = 15
FACTS_PER_CONVERSATION = [2, 3, 4]
CONVERSATION_LENGTH = 15

PROBE_TURNS = [5, 10, 15]

# Topics
TOPICS = ["dietary", "work", "hobbies"]

# Output paths
# It's better to construct these relative to the config_dir or project root
# to avoid issues if you run scripts from different directories.
# Assuming 'data' is a subfolder of the directory where config.py lives
OUTPUT_DIR = config_dir / "data" / "conversations"
CONVERSATIONS_FILE = OUTPUT_DIR / "conversations.json"
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.json"
RESULTS_FILE = OUTPUT_DIR / "results.json"

# Ensure output directory exists when config is loaded
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Convert Path objects to strings for compatibility if needed,
# though most functions handling file paths can work with Path objects.
CONVERSATIONS_FILE = str(CONVERSATIONS_FILE)
PREDICTIONS_FILE = str(PREDICTIONS_FILE)
RESULTS_FILE = str(RESULTS_FILE)