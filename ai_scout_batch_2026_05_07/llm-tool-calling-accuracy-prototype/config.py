import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

class LLMConfig:
    """
    Configuration settings for Large Language Models used in the prototype.
    API keys are loaded from environment variables for security.
    """
    # General LLM settings
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 1024
    DEFAULT_TIMEOUT_SECONDS: int = 60 # Timeout for API calls

    # Primary Agent LLM specific settings
    PRIMARY_AGENT_MODEL: str = os.getenv("PRIMARY_AGENT_MODEL", "gpt-4o-mini")

    # Validation Agent LLM specific settings
    VALIDATION_AGENT_MODEL: str = os.getenv("VALIDATION_AGENT_MODEL", "gpt-4o-mini")

    # API Keys (loaded from environment variables)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    # Add other LLM provider keys as needed, e.g., GOOGLE_API_KEY, COHERE_API_KEY

    @classmethod
    def validate_keys(cls):
        """
        Validates if necessary API keys are present for the configured models.
        Prints warnings if a model is specified but its corresponding key is missing.
        """
        if "gpt" in cls.PRIMARY_AGENT_MODEL.lower() and not cls.OPENAI_API_KEY:
            print("WARNING: OpenAI API key (OPENAI_API_KEY) not found in environment for Primary Agent model. LLM calls may fail.")
        if "claude" in cls.PRIMARY_AGENT_MODEL.lower() and not cls.ANTHROPIC_API_KEY:
            print("WARNING: Anthropic API key (ANTHROPIC_API_KEY) not found in environment for Primary Agent model. LLM calls may fail.")
        
        if "gpt" in cls.VALIDATION_AGENT_MODEL.lower() and not cls.OPENAI_API_KEY:
            print("WARNING: OpenAI API key (OPENAI_API_KEY) not found in environment for Validation Agent model. LLM calls may fail.")
        if "claude" in cls.VALIDATION_AGENT_MODEL.lower() and not cls.ANTHROPIC_API_KEY:
            print("WARNING: Anthropic API key (ANTHROPIC_API_KEY) not found in environment for Validation Agent model. LLM calls may fail.")

class PathConfig:
    """
    Configuration for file and directory paths used throughout the project.
    All paths are relative to the project's base directory.
    """
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    SRC_DIR: str = os.path.join(BASE_DIR, "src")
    AGENTS_DIR: str = os.path.join(SRC_DIR, "agents")
    TOOLS_DIR: str = os.path.join(SRC_DIR, "tools")
    VALIDATION_DIR: str = os.path.join(SRC_DIR, "validation")
    PROMPTING_DIR: str = os.path.join(SRC_DIR, "prompting")
    UTILS_DIR: str = os.path.join(SRC_DIR, "utils")

    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    EVAL_DIR: str = os.path.join(BASE_DIR, "eval")

    TOOL_DEFINITIONS_PATH: str = os.path.join(TOOLS_DIR, "tool_definitions.py")
    # Add paths for specific prompt templates if needed, e.g.,
    # PRIMARY_AGENT_PROMPT_TEMPLATE: str = os.path.join(PROMPTING_DIR, "primary_agent_prompt.txt")


class ValidationConfig:
    """
    Configuration for the multi-layered validation system.
    Defaults prioritize maximum accuracy and thorough checking.
    """
    SCHEMA_VALIDATION_ENABLED: bool = True
    HIERARCHICAL_VALIDATION_ENABLED: bool = True # Enables the orchestration of validation layers

    # Configuration for the Primary Agent's self-correction loop
    MAX_PRIMARY_AGENT_SELF_CORRECTION_ATTEMPTS: int = 2

    # Configuration for the Validation Agent's role
    VALIDATION_AGENT_DIRECT_CORRECTION_ENABLED: bool = True # If True, Validation Agent can directly output corrected call
    DEFER_TO_VALIDATION_AGENT_ON_SCHEMA_FAIL: bool = True # If True, schema validation failures immediately trigger Validation Agent review
    ALLOW_PARTIAL_SCHEMA_CORRECTION: bool = True # Allow correction mechanisms to fix only problematic fields

    # Strictness levels for schema validation (e.g., 'strict', 'lenient')
    SCHEMA_VALIDATION_STRICTNESS: str = "strict"


class LogConfig:
    """
    Configuration for application logging.
    """
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper() # INFO, DEBUG, WARNING, ERROR, CRITICAL
    LOG_FILE_PATH: str = os.path.join(PathConfig.BASE_DIR, "app.log")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class AppConfig:
    """
    General application-wide settings.
    """
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    # Placeholder for a global system message or persona for general LLM interactions
    GLOBAL_SYSTEM_MESSAGE: str = "You are a highly accurate and reliable AI assistant. Your primary goal is to achieve perfect tool calling accuracy by following all instructions and schemas precisely."


# Perform initial validation of LLM API keys on module import
LLMConfig.validate_keys()<ctrl63>