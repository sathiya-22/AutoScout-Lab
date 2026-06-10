from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Configuration settings for the Adversarial Prompt Detection Filter.
    Loads environment variables and sets default values.
    """
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    gemini_api_key: str = Field(alias='GEMINI_API_KEY')
    main_model_name: str = "gemini-1.5-flash"  # Model for generating main responses
    filter_model_name: str = "gemini-1.5-flash" # Model for classifying prompts (should be fast)
    temperature: float = 0.7                   # Creativity of the main model (0.0-1.0)
    max_tokens: int = 2048                     # Maximum output tokens for the main model
