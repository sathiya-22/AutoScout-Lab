from pydantic_settings import BaseSettings, SettingsConfigDict

class AgentConfig(BaseSettings):
    """
    Configuration settings for the AI agent and GenAI model.
    Reads environment variables prefixed with 'GEMINI_'.
    """
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 1024
    api_key: str # Automatically loaded from GEMINI_API_KEY env var

    model_config = SettingsConfigDict(env_prefix='GEMINI_',
