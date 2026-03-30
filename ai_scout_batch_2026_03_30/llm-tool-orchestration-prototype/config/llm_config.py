import os
from typing import Dict, Any, Optional

class LLMConfig:
    """
    Centralized configuration for Large Language Models (LLMs).
    Manages API keys, model parameters, and default settings for various LLM providers,
    supporting the Hierarchical Agentic System with robustness and safety in mind.
    """

    # --- LLM API Keys (loaded from environment variables for security) ---
    # These environment variables should be set in the deployment environment.
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    # Add API keys for other LLM providers as needed (e.g., COHERE_API_KEY, HUGGINGFACE_API_KEY)

    # --- Default Global Model Settings ---
    # These apply unless overridden by provider-specific or agent-specific configurations.
    DEFAULT_PROVIDER: str = "openai"  # The default LLM provider to use across the system.
    DEFAULT_MODEL: str = "gpt-4o-mini"  # The default model name for general LLM interactions.
    DEFAULT_TEMPERATURE: float = 0.5  # Controls the randomness of the LLM's output (0.0 to 1.0).
    DEFAULT_MAX_TOKENS: int = 2000  # The maximum number of tokens the LLM can generate in a response.
    DEFAULT_TIMEOUT: int = 120  # Timeout for LLM API calls in seconds.

    # --- Agent-Specific Model Overrides ---
    # Different agents might utilize different models based on their specific task requirements
    # (e.g., a more powerful model for orchestration, a faster one for verification).
    ORCHESTRATOR_MODEL: str = "gpt-4o"  # Model for the primary orchestrator, often requiring complex reasoning.
    TOOL_EXECUTOR_MODEL: str = "gpt-4o-mini"  # Model for tool execution, balancing capability and cost/speed.
    VERIFIER_MODEL: str = "gpt-3.5-turbo"  # Model for the verifier agent, prioritizing speed and reliability for checks.

    # --- Provider-specific configurations and available models ---
    # Defines specific settings, default models, and model-specific overrides for each provider.
    PROVIDER_SETTINGS: Dict[str, Dict[str, Any]] = {
        "openai": {
            "api_base": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
            "models": {
                "gpt-4o": {"max_tokens": 4096},
                "gpt-4o-mini": {"max_tokens": 2048},
                "gpt-3.5-turbo": {"max_tokens": 4096},
            }
        },
        "anthropic": {
            "api_base": "https://api.anthropic.com/v1",
            "default_model": "claude-3-sonnet-20240229",
            "models": {
                "claude-3-opus-20240229": {"max_tokens": 4096},
                "claude-3-sonnet-20240229": {"max_tokens": 2048},
                "claude-3-haiku-20240307": {"max_tokens": 1024},
            }
        },
        "google": {
            "api_base": "https://generativelanguage.googleapis.com/v1beta",
            "default_model": "gemini-pro",
            "models": {
                "gemini-pro": {"max_tokens": 2048},
            }
        },
        "local_ollama": {
            "api_base": "http://localhost:11434/v1",
            "default_model": "llama3",
            "models": {
                "llama3": {"max_tokens": 4096},
                "mistral": {"max_tokens": 2048},
            }
        }
        # Add more providers as necessary
    }

    @staticmethod
    def get_api_key(provider: str) -> str:
        """
        Retrieves the API key for a specified LLM provider.
        Raises a ValueError if the key is not found for non-local providers.
        """
        provider_key_map = {
            "openai": LLMConfig.OPENAI_API_KEY,
            "anthropic": LLMConfig.ANTHROPIC_API_KEY,
            "google": LLMConfig.GOOGLE_API_KEY,
            # Extend with other providers
        }
        key = provider_key_map.get(provider.lower())

        if not key:
            # Local Ollama typically does not require an API key
            if provider.lower() == "local_ollama":
                return ""
            raise ValueError(
                f"API key for provider '{provider}' is not configured. "
                f"Please set the '{provider.upper()}_API_KEY' environment variable."
            )
        return key

    @staticmethod
    def get_default_model(provider: str) -> str:
        """
        Retrieves the default model name for a given LLM provider.
        Falls back to the global DEFAULT_MODEL if not specified for the provider.
        """
        provider_config = LLMConfig.PROVIDER_SETTINGS.get(provider.lower())
        if provider_config and "default_model" in provider_config:
            return provider_config["default_model"]
        return LLMConfig.DEFAULT_MODEL

    @staticmethod
    def get_model_setting(
        provider: str, model_name: str, setting: str, default_value: Any
    ) -> Any:
        """
        Retrieves a specific setting (e.g., 'max_tokens', 'temperature') for a given model
        from a specified provider. Prioritizes model-specific settings, then provider defaults
        (if implemented), and finally falls back to a global default value.
        """
        provider_config = LLMConfig.PROVIDER_SETTINGS.get(provider.lower(), {})
        model_config = provider_config.get("models", {}).get(model_name.lower(), {})

        if setting in model_config:
            return model_config[setting]
        
        # Consider adding provider-level default settings here if they apply to all models
        # of that provider unless overridden by specific models.

        return default_value

    @staticmethod
    def validate_config():
        """
        Performs basic validation on essential configuration parameters to ensure the system
        can operate correctly. Raises ValueError for critical missing or invalid settings.
        """
        if not LLMConfig.DEFAULT_MODEL:
            raise ValueError("LLMConfig.DEFAULT_MODEL must be defined.")
        if not isinstance(LLMConfig.DEFAULT_TEMPERATURE, (int, float)):
            raise ValueError("LLMConfig.DEFAULT_TEMPERATURE must be a numeric value.")
        if not isinstance(LLMConfig.DEFAULT_MAX_TOKENS, int) or LLMConfig.DEFAULT_MAX_TOKENS <= 0:
            raise ValueError("LLMConfig.DEFAULT_MAX_TOKENS must be a positive integer.")
        if not isinstance(LLMConfig.DEFAULT_TIMEOUT, int) or LLMConfig.DEFAULT_TIMEOUT <= 0:
            raise ValueError("LLMConfig.DEFAULT_TIMEOUT must be a positive integer.")

        # Check API key for the default provider, unless it's a local model
        if LLMConfig.DEFAULT_PROVIDER.lower() not in ["local_ollama"]:
            try:
                LLMConfig.get_api_key(LLMConfig.DEFAULT_PROVIDER)
            except ValueError as e:
                # Issue a warning rather than a hard error during config validation,
                # as specific agents might override the provider later.
                print(f"Warning: Default LLM provider '{LLMConfig.DEFAULT_PROVIDER}' selected, "
                      f"but its API key is missing. Error: {e}")<ctrl63>