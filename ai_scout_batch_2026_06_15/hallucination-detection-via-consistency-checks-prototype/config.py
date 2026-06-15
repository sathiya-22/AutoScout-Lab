from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Config(BaseSettings):
    """
    Configuration settings for the Hallucination Detection prototype.
    Settings are loaded from environment variables or a .env file.
    """
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_tokens: int = 500
    api_key: str = Field(..., env="GEMINI_API_KEY") # Reads from GEMINI_API_KEY environment variable

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
