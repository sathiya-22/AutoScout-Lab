from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Configuration settings for the Retrieval Confidence Calibration Module.
    Settings are loaded from environment variables or a .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: SecretStr = Field(..., env="GEMINI_API_KEY", description="Your Google Gemini API key.")
    model_name: str = Field("gemini-1.5-flash", description="The name of the Gemini model to use.")
    temperature: float = Field(0.2, description="The sampling temperature for text generation. Lower values make the output more deterministic.")
    max_tokens: int = Field(500, description="The maximum number of tokens to generate in the output.")
