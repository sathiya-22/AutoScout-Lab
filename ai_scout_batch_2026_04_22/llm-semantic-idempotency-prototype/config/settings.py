import os
from dotenv import load_dotenv

load_dotenv()

class LLMProviderSettings:
    """
    Configuration for interacting with Language Model Providers.
    """
    DEFAULT_PROVIDER = os.getenv("LLM_DEFAULT_PROVIDER", "openai")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") # Can be used for custom endpoints
    OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60")) # seconds

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-haiku-20240307")
    ANTHROPIC_TIMEOUT = int(os.getenv("ANTHROPIC_TIMEOUT", "60"))

    # Add more providers as needed (e.g., Gemini, local LLMs)

    @classmethod
    def get_api_key(cls, provider: str = DEFAULT_PROVIDER):
        key_map = {
            "openai": cls.OPENAI_API_KEY,
            "anthropic": cls.ANTHROPIC_API_KEY,
        }
        key = key_map.get(provider.lower())
        if not key:
            raise ValueError(f"API key for LLM provider '{provider}' not found or configured.")
        return key

    @classmethod
    def get_model_name(cls, provider: str = DEFAULT_PROVIDER):
        model_map = {
            "openai": cls.OPENAI_MODEL_NAME,
            "anthropic": cls.ANTHROPIC_MODEL_NAME,
        }
        model = model_map.get(provider.lower())
        if not model:
            raise ValueError(f"Model name for LLM provider '{provider}' not found or configured.")
        return model

    @classmethod
    def get_timeout(cls, provider: str = DEFAULT_PROVIDER):
        timeout_map = {
            "openai": cls.OPENAI_TIMEOUT,
            "anthropic": cls.ANTHROPIC_TIMEOUT,
        }
        timeout = timeout_map.get(provider.lower())
        if timeout is None:
            raise ValueError(f"Timeout for LLM provider '{provider}' not found or configured.")
        return timeout

class AgentSettings:
    """
    General settings for agents.
    """
    DEFAULT_RETRY_ATTEMPTS = int(os.getenv("AGENT_DEFAULT_RETRY_ATTEMPTS", "3"))
    DEFAULT_RETRY_DELAY_SECONDS = int(os.getenv("AGENT_DEFAULT_RETRY_DELAY_SECONDS", "5"))
    AGENT_WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "agent_work")

    @classmethod
    def initialize_agent_working_directory(cls):
        """Ensures the agent working directory exists."""
        os.makedirs(cls.AGENT_WORKING_DIR, exist_ok=True)


class SemanticReconciliationSettings:
    """
    Configuration for the Semantic Reconciliation Layer.
    """
    DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SEMREC_DEFAULT_SIMILARITY_THRESHOLD", "0.8")) # For embedding-based comparison
    KEYWORD_MATCH_MIN_RATIO = float(os.getenv("SEMREC_KEYWORD_MATCH_MIN_RATIO", "0.6")) # For keyword-based comparison
    JSON_SCHEMA_VALIDATION_STRICT = os.getenv("SEMREC_JSON_SCHEMA_VALIDATION_STRICT", "True").lower() == "true"
    # Placeholder for embedding model if different from main LLM
    EMBEDDING_MODEL_NAME = os.getenv("SEMREC_EMBEDDING_MODEL_NAME", "text-embedding-3-small")


class ValidationGateSettings:
    """
    Configuration for validation gates.
    """
    MAX_VALIDATION_RETRIES = int(os.getenv("VALIDATION_MAX_RETRIES", "2"))
    VALIDATION_FAILURE_MODE = os.getenv("VALIDATION_FAILURE_MODE", "CRASH") # Options: CRASH, LOG_AND_CONTINUE (for non-critical)
    REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "validation")

    @classmethod
    def initialize_report_directory(cls):
        """Ensures the validation report directory exists."""
        os.makedirs(cls.REPORT_DIR, exist_ok=True)


class DeterministicLLMProxySettings:
    """
    Configuration for deterministic LLM invocation proxies.
    """
    DEFAULT_SEED_VALUE = int(os.getenv("PROXY_DEFAULT_SEED_VALUE", "42")) # For seeded_proxy
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "llm_responses")
    CACHE_EXPIRATION_SECONDS = int(os.getenv("PROXY_CACHE_EXPIRATION_SECONDS", "3600")) # 1 hour
    CACHE_ENABLED = os.getenv("PROXY_CACHE_ENABLED", "True").lower() == "true"

    @classmethod
    def initialize_cache_directory(cls):
        """Ensures the cache directory exists."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


class LoggingSettings:
    """
    Configuration for logging.
    """
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "system.log")
    LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760")) # 10 MB
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    @classmethod
    def initialize_log_directory(cls):
        """Ensures the log directory exists."""
        log_dir = os.path.dirname(cls.LOG_FILE_PATH)
        os.makedirs(log_dir, exist_ok=True)


class AppSettings:
    """
    General application settings.
    """
    ENV = os.getenv("APP_ENV", "development")
    DEBUG = os.getenv("APP_DEBUG", "True").lower() == "true"
    PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    @classmethod
    def initialize_directories(cls):
        """Initialize all necessary application directories."""
        AgentSettings.initialize_agent_working_directory()
        ValidationGateSettings.initialize_report_directory()
        DeterministicLLMProxySettings.initialize_cache_directory()
        LoggingSettings.initialize_log_directory()

# Initialize directories on import
AppSettings.initialize_directories()

# Example of how to access settings:
# from config.settings import LLMProviderSettings
# api_key = LLMProviderSettings.get_api_key("openai")
# cache_dir = DeterministicLLMProxySettings.CACHE_DIR