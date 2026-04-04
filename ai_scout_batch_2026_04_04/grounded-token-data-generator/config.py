```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This allows sensitive information and environment-specific settings to be managed outside the codebase.
load_dotenv()

class Config:
    """
    Centralized configuration settings for the Grounded Token Data Generator project.
    Settings are loaded from environment variables (via .env) or use default values.
    """

    # --- Project Metadata ---
    PROJECT_NAME = "Grounded Token Data Generator"
    VERSION = "0.1.0"

    # --- Paths and Data Management ---
    # BASE_DIR is assumed to be the project root where config.py resides.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    DATA_DIR = os.path.join(BASE_DIR, "src", "data")
    INPUT_DIR = os.path.join(DATA_DIR, "input")
    OUTPUT_DIR = os.path.join(DATA_DIR, "output")

    NEW_TOKENS_FILE = os.path.join(INPUT_DIR, "new_tokens.csv")
    GENERATED_SUPERVISION_FILE = os.path.join(OUTPUT_DIR, "generated_supervision.json")
    
    # Ensure necessary directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- LLM Integration Settings ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo-preview")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.7))
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 500))
    # Timeout for OpenAI API calls in seconds
    OPENAI_REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", 60))

    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    HUGGINGFACE_MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
    # Base URL for Hugging Face Inference API (if self-hosted or specific endpoint)
    HUGGINGFACE_API_BASE = os.getenv("HUGGINGFACE_API_BASE", "https://api-inference.huggingface.co/models/")
    
    # Default LLM provider to use if not specified otherwise
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()
    
    # --- Knowledge Integration Settings ---
    WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    # Path to a local knowledge base (e.g., SQLite database for structured data)
    LOCAL_KNOWLEDGE_DB_PATH = os.path.join(DATA_DIR, "knowledge_base.db")
    # Comma-separated list of knowledge sources to prioritize or use by default
    DEFAULT_KNOWLEDGE_SOURCES_STR = os.getenv("DEFAULT_KNOWLEDGE_SOURCES", "wikidata,local_db")
    DEFAULT_KNOWLEDGE_SOURCES = [
        s.strip() for s in DEFAULT_KNOWLEDGE_SOURCES_STR.split(',') if s.strip()
    ]

    # --- Quality Assessment Settings ---
    QUALITY_THRESHOLD_ACCEPT = float(os.getenv("QUALITY_THRESHOLD_ACCEPT", 0.75)) # Score above this considered high quality
    QUALITY_THRESHOLD_REVIEW = float(os.getenv("QUALITY_THRESHOLD_REVIEW", 0.40)) # Score between review and accept needs review
    QUALITY_THRESHOLD_REJECT = float(os.getenv("QUALITY_THRESHOLD_REJECT", 0.20)) # Score below this considered low quality
    
    # Embedding model for semantic similarity metrics
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Maximum length for generated descriptions (for truncation or LLM generation control)
    MAX_DESCRIPTION_LENGTH = int(os.getenv("MAX_DESCRIPTION_LENGTH", 200))

    # --- Human-in-the-Loop (HIL) & Active Learning Settings ---
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Number of candidate descriptions to present to a human annotator in one batch
    BATCH_SIZE_HIL = int(os.getenv("BATCH_SIZE_HIL", 10))
    
    # Strategy for active learning: 'uncertainty', 'diversity', 'random'
    ACTIVE_LEARNING_STRATEGY = os.getenv("ACTIVE_LEARNING_STRATEGY", "uncertainty").lower()
    
    # Percentage of data to be sampled for active learning initial pool
    ACTIVE_LEARNING_INITIAL_SAMPLE_RATE = float(os.getenv("ACTIVE_LEARNING_INITIAL_SAMPLE_RATE", 0.1))

    # --- General Utilities & Error Handling ---
    MAX_API_RETRIES = int(os.getenv("MAX_API_RETRIES", 3))
    API_RETRY_DELAY_SECONDS = int(os.getenv("API_RETRY_DELAY_SECONDS", 5)) # Exponential backoff recommended in practice

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE = os.path.join(BASE_DIR, "app.log")


# Instantiate the configuration object for easy import
settings = Config()
```