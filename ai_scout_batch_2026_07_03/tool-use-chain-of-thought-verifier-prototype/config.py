from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings for the AI model and API key.
    Reads environment variables for configuration.
    """
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_tokens: int = 1024
    api_key: str = Field(..., env="GEMINI_API_KEY") # Reads from GEMINI_API_KEY env var

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Create a singleton instance of settings to be imported across the project
settings = Settings()
