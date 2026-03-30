import os
from typing import List, Literal, Optional

class Settings:
    # --- LLM Provider Settings ---
    LLM_PROVIDER: Literal["openai", "anthropic", "google", "local"] = os.getenv("LLM_PROVIDER", "openai").lower()

    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY") # Example for other providers

    # --- Default LLM Model Parameters ---
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4o")
    VERIFIER_MODEL: str = os.getenv("VERIFIER_MODEL", "gpt-3.5-turbo") # Often a smaller, faster model
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))
    TOP_P: float = float(os.getenv("TOP_P", "1.0"))
    PRESENCE_PENALTY: float = float(os.getenv("PRESENCE_PENALTY", "0.0"))
    FREQUENCY_PENALTY: float = float(os.getenv("FREQUENCY_PENALTY", "0.0"))

    # --- Agent Behavior Settings ---
    VERIFICATION_REQUIRED_FOR_TOOLS: bool = os.getenv("VERIFICATION_REQUIRED_FOR_TOOLS", "True").lower() == "true"
    MAX_REPROMPT_ATTEMPTS: int = int(os.getenv("MAX_REPROMPT_ATTEMPTS", "3"))
    AGENT_TIMEOUT_SECONDS: int = int(os.getenv("AGENT_TIMEOUT_SECONDS", "300")) # Overall task timeout
    TOOL_EXECUTION_TIMEOUT_SECONDS: int = int(os.getenv("TOOL_EXECUTION_TIMEOUT_SECONDS", "60"))

    # --- Tool Execution Settings ---
    SANDBOX_ENABLED: bool = os.getenv("SANDBOX_ENABLED", "True").lower() == "true"
    SANDBOX_IMAGE: str = os.getenv("SANDBOX_IMAGE", "python:3.10-slim-buster") # For Docker-based sandbox
    SANDBOX_WORK_DIR: str = os.getenv("SANDBOX_WORK_DIR", "/app/sandbox")
    ALLOWED_TOOL_CATEGORIES: List[str] = os.getenv(
        "ALLOWED_TOOL_CATEGORIES", "file_system,internet_access,data_processing"
    ).split(',')

    # --- Guardrail Settings ---
    PII_DETECTION_ENABLED: bool = os.getenv("PII_DETECTION_ENABLED", "True").lower() == "true"
    SQL_INJECTION_SCAN_ENABLED: bool = os.getenv("SQL_INJECTION_SCAN_ENABLED", "True").lower() == "true"
    MALICIOUS_CODE_SCAN_ENABLED: bool = os.getenv("MALICIOUS_CODE_SCAN_ENABLED", "True").lower() == "true"
    MAX_OUTPUT_LENGTH_CHARS: int = int(os.getenv("MAX_OUTPUT_LENGTH_CHARS", "10000")) # Post-execution output validation
    SAFETY_SCORE_THRESHOLD: float = float(os.getenv("SAFETY_SCORE_THRESHOLD", "0.8")) # For AI-based guardrails

    # --- Logging Settings ---
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- Internal Paths (relative to project root usually) ---
    TOOL_DEFINITION_PATH: str = "tools/tool_definitions.py"
    PROMPT_TEMPLATE_PATH: str = "llm_integrations/prompt_templates/"
    GUARDRAILS_CONFIG_PATH: str = "validation/guardrail_rules.json" # Example for external rules

    def __post_init__(self):
        self._validate_settings()

    def _validate_settings(self):
        """Perform basic validation and error handling for critical settings."""
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set for OpenAI provider.")
        if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY must be set for Anthropic provider.")
        if self.LLM_PROVIDER == "google" and not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY must be set for Google provider.")
        # Add checks for other providers if necessary
        
        if not (0.0 <= self.TEMPERATURE <= 2.0):
            raise ValueError("TEMPERATURE must be between 0.0 and 2.0.")
        if not (0.0 <= self.TOP_P <= 1.0):
            raise ValueError("TOP_P must be between 0.0 and 1.0.")
        if self.MAX_TOKENS <= 0:
            raise ValueError("MAX_TOKENS must be a positive integer.")
        if self.MAX_REPROMPT_ATTEMPTS <= 0:
            raise ValueError("MAX_REPROMPT_ATTEMPTS must be a positive integer.")

# Instantiate settings to be imported
settings = Settings()

# Optional: Load environment variables from a .env file if dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
    # Re-instantiate settings after loading .env to pick up new values
    settings = Settings()
except ImportError:
    pass # dotenv not installed, proceed without it