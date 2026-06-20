from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Config(BaseSettings):
    """
    Configuration settings for the Agentic Code Review Assistant.
    Settings are loaded from environment variables.
    """
    model_config = SettingsConfigDict(env_file='.env', extra='ignore') # Optional: load from .env file

    api_key: str = Field(..., env='GEMINI_API_KEY')
    model_name: str = "gemini-1.5-flash" # Using gemini-1.5-flash as default, as gemini-2.5-flash is not a standard model name
    temperature: float = 0.2
    max_tokens: int = 2048
