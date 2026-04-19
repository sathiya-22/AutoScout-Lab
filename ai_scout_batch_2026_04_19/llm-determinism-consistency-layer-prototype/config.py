import os
import logging

class Config:
    """
    Configuration settings for the LLM Determinism and Consistency Layer.
    Manages parameters for LLMs, thresholds, and data paths.
    """

    # --- General Settings ---
    PROJECT_NAME: str = "LLM Determinism Layer"
    DEFAULT_LOG_LEVEL: int = logging.INFO
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "logs/llm_determinism.log")
    
    # --- LLM Provider Settings ---
    # Primary LLM for general operations
    PRIMARY_LLM_PROVIDER: str = os.getenv("PRIMARY_LLM_PROVIDER", "openai")
    PRIMARY_LLM_MODEL: str = os.getenv("PRIMARY_LLM_MODEL", "gpt-4o-mini")
    PRIMARY_LLM_API_KEY: str = os.getenv("PRIMARY_LLM_API_KEY") # Recommended to set via env var
    PRIMARY_LLM_BASE_URL: str = os.getenv("PRIMARY_LLM_BASE_URL", None)
    PRIMARY_LLM_TIMEOUT: int = int(os.getenv("PRIMARY_LLM_TIMEOUT", 60))
    PRIMARY_LLM_TEMPERATURE: float = float(os.getenv("PRIMARY_LLM_TEMPERATURE", 0.1)) # Lower for determinism

    # Secondary LLM for fallback or specific tasks (e.g., simpler, cheaper)
    FALLBACK_LLM_PROVIDER: str = os.getenv("FALLBACK_LLM_PROVIDER", "openai")
    FALLBACK_LLM_MODEL: str = os.getenv("FALLBACK_LLM_MODEL", "gpt-3.5-turbo")
    FALLBACK_LLM_API_KEY: str = os.getenv("FALLBACK_LLM_API_KEY") # Recommended to set via env var
    FALLBACK_LLM_BASE_URL: str = os.getenv("FALLBACK_LLM_BASE_URL", None)
    FALLBACK_LLM_TIMEOUT: int = int(os.getenv("FALLBACK_LLM_TIMEOUT", 30))
    FALLBACK_LLM_TEMPERATURE: float = float(os.getenv("FALLBACK_LLM_TEMPERATURE", 0.3))

    # --- Determinism Layer Feature Toggles ---
    ENABLE_VALIDATION: bool = os.getenv("ENABLE_VALIDATION", "True").lower() == "true"
    ENABLE_ANCHORING: bool = os.getenv("ENABLE_ANCHORING", "True").lower() == "true"
    ENABLE_RELIABILITY_SCORING: bool = os.getenv("ENABLE_RELIABILITY_SCORING", "True").lower() == "true"
    ENABLE_AGENT_TEST_HARNESS: bool = os.getenv("ENABLE_AGENT_TEST_HARNESS", "True").lower() == "true"

    # --- Validation & Correction Settings ---
    MAX_REPROMPTS_FOR_CORRECTION: int = int(os.getenv("MAX_REPROMPTS_FOR_CORRECTION", 3))
    VALIDATION_SCHEMA_DIR: str = os.getenv("VALIDATION_SCHEMA_DIR", "data/schemas")
    GROUND_TRUTH_DATA_DIR: str = os.getenv("GROUND_TRUTH_DATA_DIR", "data/ground_truth")
    # Strategy for validation failures: "reprompt", "self_correct", "fallback", "human_review"
    CORRECTION_STRATEGY: str = os.getenv("CORRECTION_STRATEGY", "reprompt")

    # --- Contextual Anchoring Settings ---
    CONTEXT_ANCHOR_PREFIX: str = os.getenv("CONTEXT_ANCHOR_PREFIX", "[ANCHOR_ID:")
    CONTEXT_ANCHOR_SUFFIX: str = os.getenv("CONTEXT_ANCHOR_SUFFIX", "]")
    STATE_MANAGER_DIR: str = os.getenv("STATE_MANAGER_DIR", "data/agent_states")
    
    # --- Reliability Scoring Settings ---
    RELIABILITY_SCORE_THRESHOLD: float = float(os.getenv("RELIABILITY_SCORE_THRESHOLD", 0.75))
    # Fallback actions: "fallback_llm", "human_review", "abort_with_error", "safe_response"
    LOW_RELIABILITY_FALLBACK_ACTION: str = os.getenv("LOW_RELIABILITY_FALLBACK_ACTION", "fallback_llm")
    
    # --- Agentic Development Settings ---
    AGENT_HARNESS_LOG_DIR: str = os.getenv("AGENT_HARNESS_LOG_DIR", "logs/agent_harness")
    AGENT_TEST_CASE_DIR: str = os.getenv("AGENT_TEST_CASE_DIR", "tests/agent_tests")

    # --- Data Paths ---
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    EXAMPLE_DATA_DIR: str = os.getenv("EXAMPLE_DATA_DIR", "examples/data")

    @classmethod
    def initialize_directories(cls):
        """Ensures all necessary directories exist."""
        directories = [
            cls.LOG_FILE_PATH,
            cls.VALIDATION_SCHEMA_DIR,
            cls.GROUND_TRUTH_DATA_DIR,
            cls.STATE_MANAGER_DIR,
            cls.AGENT_HARNESS_LOG_DIR,
            cls.AGENT_TEST_CASE_DIR,
            cls.DATA_DIR,
            cls.EXAMPLE_DATA_DIR
        ]
        
        # Extract unique parent directories
        unique_dirs = set()
        for path in directories:
            if "." in os.path.basename(path): # It's a file path, get its directory
                unique_dirs.add(os.path.dirname(path))
            else: # It's already a directory
                unique_dirs.add(path)

        for d in unique_dirs:
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
                logging.debug(f"Created directory: {d}")

    @classmethod
    def validate_config(cls):
        """Performs basic validation of configuration settings."""
        if not cls.PRIMARY_LLM_API_KEY:
            logging.warning("PRIMARY_LLM_API_KEY is not set. LLM calls may fail.")
        if cls.ENABLE_RELIABILITY_SCORING and not (0 <= cls.RELIABILITY_SCORE_THRESHOLD <= 1):
            raise ValueError("RELIABILITY_SCORE_THRESHOLD must be between 0 and 1.")
        
        # Check if directories can be created/accessed
        try:
            cls.initialize_directories()
        except OSError as e:
            logging.error(f"Error creating necessary directories: {e}")
            raise

# Initialize directories on import
Config.initialize_directories()
Config.validate_config()

# Configure basic logging
logging.basicConfig(level=Config.DEFAULT_LOG_LEVEL,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(Config.LOG_FILE_PATH),
                        logging.StreamHandler()
                    ])

# Example of how to set up more specific logging for the determinism layer itself
logging.getLogger(__name__).setLevel(Config.DEFAULT_LOG_LEVEL)

# Suppress warnings from libraries if needed, e.g., for missing API keys in tests
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("openai").setLevel(logging.WARNING)
# logging.getLogger("anthropic").setLevel(logging.WARNING)