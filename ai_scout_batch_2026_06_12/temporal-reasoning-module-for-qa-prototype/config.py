from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Config(BaseSettings):
    """
    Configuration settings for the Temporal Reasoning Module.
    Loads environment variables and sets default model parameters.
    """
    model_config = SettingsConfigDict(
        env_file='.env',         # Load environment variables from a .env file
        extra='ignore'           # Ignore extra environment variables not defined here
    )

    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.4
    max_tokens: int = 500
    api_key: str = Field(..., env="GEMINI_API_KEY") # Reads GEMINI_API_KEY from environment
