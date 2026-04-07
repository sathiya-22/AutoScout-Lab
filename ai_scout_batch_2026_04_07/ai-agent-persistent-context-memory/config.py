import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # --- General Application Settings ---
    APP_NAME: str = "AI Agent Prototype"
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper() # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "agent_log.log")

    # --- LLM API Settings ---
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "YOUR_OPENAI_API_KEY") # Placeholder, ensure secure handling
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai") # e.g., "openai", "anthropic", "huggingface"
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 2048))
    LLM_API_BASE_URL: str = os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1") # For custom endpoints

    # --- Database Settings (Persistence Layer) ---
    # SQLite is used for prototyping. For production, consider PostgreSQL, MySQL, etc.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./agent_memory.db")

    # Vector Store Settings
    VECTOR_STORE_PROVIDER: str = os.getenv("VECTOR_STORE_PROVIDER", "chroma") # e.g., "chroma", "pinecone", "qdrant"
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    # PINE_CONE_API_KEY: str = os.getenv("PINE_CONE_API_KEY", "") # Example for Pinecone
    # PINE_CONE_ENVIRONMENT: str = os.getenv("PINE_CONE_ENVIRONMENT", "")
    # PINE_CONE_INDEX_NAME: str = os.getenv("PINE_CONE_INDEX_NAME", "agent-memory")

    # --- File Paths/Directories ---
    PROMPTS_DIR: str = os.getenv("PROMPTS_DIR", "prompts")
    GENERATED_SKILLS_DIR: str = os.getenv("GENERATED_SKILLS_DIR", "skills/generated_skills")
    # Ensure these directories exist or are created at runtime
    PERSISTENCE_DIR: str = os.getenv("PERSISTENCE_DIR", "./data") # Generic directory for data persistence

    # --- Context Management Settings ---
    DEFAULT_CONTEXT_WINDOW_SIZE: int = int(os.getenv("DEFAULT_CONTEXT_WINDOW_SIZE", 8000)) # Max tokens for LLM context
    MAX_CONTEXT_HISTORY_LENGTH: int = int(os.getenv("MAX_CONTEXT_HISTORY_LENGTH", 10)) # Max recent turns to include
    MAX_SEMANTIC_RETRIEVAL_RESULTS: int = int(os.getenv("MAX_SEMANTIC_RETRIEVAL_RESULTS", 5))
    SEMANTIC_SIMILARITY_THRESHOLD: float = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", 0.75))
    CHANGELOG_MAX_ENTRIES: int = int(os.getenv("CHANGELOG_MAX_ENTRIES", 10))
    TREE_CONTEXT_MAX_DEPTH: int = int(os.getenv("TREE_CONTEXT_MAX_DEPTH", 3)) # Max depth for tree context traversal

    # --- Checkpoint Manager Settings ---
    CHECKPOINT_FREQUENCY_TURNS: int = int(os.getenv("CHECKPOINT_FREQUENCY_TURNS", 5)) # Create checkpoint every N turns
    CHECKPOINT_MIN_PROGRESS_THRESHOLD: float = float(os.getenv("CHECKPOINT_MIN_PROGRESS_THRESHOLD", 0.1)) # Min progress before checkpoint

    # --- Knowledge Crystallization Settings ---
    CRYSTALLIZATION_TRIGGER_SUCCESS_COUNT: int = int(os.getenv("CRYSTALLIZATION_TRIGGER_SUCCESS_COUNT", 3)) # Trigger after N successful tasks
    CRYSTALLIZATION_COOLDOWN_TURNS: int = int(os.getenv("CRYSTALLIZATION_COOLDOWN_TURNS", 10)) # Cooldown period for crystallization
    SKILL_FILE_FORMAT: str = os.getenv("SKILL_FILE_FORMAT", "py") # e.g., "py", "json"

    # --- Context Optimization Settings ---
    OPTIMIZER_LEARNING_RATE: float = float(os.getenv("OPTIMIZER_LEARNING_RATE", 0.05))
    CONTEXT_SOURCE_WEIGHTS: dict = { # Initial weights for context sources (can be optimized)
        "semantic_retrieval": float(os.getenv("CONTEXT_WEIGHT_SEMANTIC", 0.4)),
        "changelog": float(os.getenv("CONTEXT_WEIGHT_CHANGELOG", 0.3)),
        "tree_context": float(os.getenv("CONTEXT_WEIGHT_TREE", 0.3)),
    }

# Instantiate settings
settings = Settings()

# Basic validation and directory creation
if not os.path.exists(settings.PROMPTS_DIR):
    os.makedirs(settings.PROMPTS_DIR, exist_ok=True)
if not os.path.exists(settings.GENERATED_SKILLS_DIR):
    os.makedirs(settings.GENERATED_SKILLS_DIR, exist_ok=True)
if not os.path.exists(settings.PERSISTENCE_DIR):
    os.makedirs(settings.PERSISTENCE_DIR, exist_ok=True)

# Example of how to access a setting
# print(f"LLM Model: {settings.LLM_MODEL_NAME}")
# print(f"Database URL: {settings.DATABASE_URL}")