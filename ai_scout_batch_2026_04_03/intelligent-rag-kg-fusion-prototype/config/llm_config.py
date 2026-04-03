import os

class LLMConfig:
    """
    Configuration settings for Large Language Models (LLMs) used across the system.
    This includes model names, API keys, and model-specific parameters.
    """

    # --- Default LLM Settings ---
    # The default LLM to be used if no specific model is specified for a task.
    DEFAULT_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")

    # --- API Keys (loaded from environment variables for security) ---
    API_KEYS: dict[str, str | None] = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
        # Add other LLM provider API keys as needed
    }

    # --- Model Provider Mapping ---
    # Maps internal model names to their respective API providers.
    # This helps llm_interface.py route requests to the correct client.
    MODEL_PROVIDERS: dict[str, str] = {
        "gpt-4o": "openai",
        "gpt-4-turbo": "openai",
        "gpt-3.5-turbo": "openai",
        "claude-3-opus-20240229": "anthropic",
        "claude-3-sonnet-20240229": "anthropic",
        "claude-3-haiku-20240229": "anthropic",
        "gemini-pro": "google",
        "mistral-large-latest": "mistral",
        "mixtral-8x7b-instruct-v0.1": "mistral",
        # Add other model mappings as needed
    }

    # --- Task-Specific LLM Models (can override DEFAULT_MODEL) ---
    # Model specifically for the Agentic Query Refinement.
    AGENT_MODEL: str = os.getenv("AGENT_LLM_MODEL", "gpt-4o")
    # Model specifically for the final response generation.
    GENERATION_MODEL: str = os.getenv("GENERATION_LLM_MODEL", "gpt-4o")

    # --- Default Model Parameters ---
    # Common parameters applicable to most LLMs.
    # Specific models or tasks might override these.
    DEFAULT_MODEL_PARAMS: dict = {
        "temperature": float(os.getenv("LLM_TEMPERATURE", 0.3)),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", 1500)),
        "top_p": float(os.getenv("LLM_TOP_P", 1.0)),
        "frequency_penalty": float(os.getenv("LLM_FREQUENCY_PENALTY", 0.0)),
        "presence_penalty": float(os.getenv("LLM_PRESENCE_PENALTY", 0.0)),
    }

    # --- Task-Specific Model Parameters (can override DEFAULT_MODEL_PARAMS) ---
    AGENT_MODEL_PARAMS: dict = {
        "temperature": float(os.getenv("AGENT_LLM_TEMPERATURE", 0.2)), # Agents might prefer lower temperature for more deterministic actions
        "max_tokens": int(os.getenv("AGENT_LLM_MAX_TOKENS", 700)),
    }
    GENERATION_MODEL_PARAMS: dict = {
        "temperature": float(os.getenv("GENERATION_LLM_TEMPERATURE", 0.4)), # Generation might prefer slightly higher temperature for creativity
        "max_tokens": int(os.getenv("GENERATION_LLM_MAX_TOKENS", 1500)),
    }

    @staticmethod
    def get_api_key(provider: str) -> str:
        """
        Retrieves the API key for a given provider.
        Raises a ValueError if the key is not found or not set in environment variables.
        """
        provider_upper = f"{provider.upper()}_API_KEY"
        api_key = LLMConfig.API_KEYS.get(provider_upper)
        if not api_key:
            raise ValueError(
                f"API key for {provider} not found. Please set the {provider_upper} "
                "environment variable or check config/llm_config.py."
            )
        return api_key

    @staticmethod
    def get_model_provider(model_name: str) -> str:
        """
        Retrieves the provider name for a given LLM model.
        Raises a ValueError if the model is not mapped to a provider.
        """
        provider = LLMConfig.MODEL_PROVIDERS.get(model_name)
        if not provider:
            raise ValueError(
                f"Model '{model_name}' not mapped to any provider. "
                "Please add it to LLMConfig.MODEL_PROVIDERS in config/llm_config.py."
            )
        return provider

    @staticmethod
    def get_model_params(task_type: str = "default") -> dict:
        """
        Retrieves model parameters for a specific task type, merging with defaults.
        Task types can be 'default', 'agent', 'generation'.
        """
        params = LLMConfig.DEFAULT_MODEL_PARAMS.copy()
        if task_type == "agent":
            params.update(LLMConfig.AGENT_MODEL_PARAMS)
        elif task_type == "generation":
            params.update(LLMConfig.GENERATION_MODEL_PARAMS)
        return params

    @staticmethod
    def get_model_name(task_type: str = "default") -> str:
        """
        Retrieves the model name for a specific task type.
        Task types can be 'default', 'agent', 'generation'.
        """
        if task_type == "agent":
            return LLMConfig.AGENT_MODEL
        elif task_type == "generation":
            return LLMConfig.GENERATION_MODEL
        else:
            return LLMConfig.DEFAULT_MODEL

# Example of how to access these configurations (for testing/debugging purposes)
if __name__ == "__main__":
    print(f"Default LLM Model: {LLMConfig.DEFAULT_MODEL}")
    print(f"Agent LLM Model: {LLMConfig.AGENT_MODEL}")
    print(f"Generation LLM Model: {LLMConfig.GENERATION_MODEL}")
    print(f"Default Model Params: {LLMConfig.get_model_params()}")
    print(f"Agent Model Params: {LLMConfig.get_model_params('agent')}")
    print(f"Generation Model Params: {LLMConfig.get_model_params('generation')}")

    try:
        # This will raise an error if OPENAI_API_KEY is not set
        openai_key = LLMConfig.get_api_key("openai")
        print(f"OpenAI API Key (first 5 chars): {openai_key[:5]}...")
    except ValueError as e:
        print(f"Error getting OpenAI API key: {e}")

    try:
        provider = LLMConfig.get_model_provider("gpt-4o")
        print(f"Provider for gpt-4o: {provider}")
        # Test an unmapped model
        LLMConfig.get_model_provider("unmapped-model")
    except ValueError as e:
        print(f"Error getting model provider: {e}")