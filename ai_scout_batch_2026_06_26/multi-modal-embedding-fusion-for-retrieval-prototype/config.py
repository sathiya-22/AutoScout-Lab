```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Configuration settings for the Multi-Modal Embedding Fusion project.
    Reads environment variables, primarily from a .env file.
    """
    model_name: str = "gemini-2.5-flash"  # Model used for multi-modal embedding
    temperature: float = 0.0             # Not directly used for embedding, but common for generative models
    max_tokens: int = 100                # Not directly used for embedding, but common for generative models
    api_key: str = Field(validation_alias="GEMINI_API_KEY") # Google Gemini API Key

    model_config = SettingsConfigDict(
        env_file='.env',         # Load environment variables from a .env file
        extra='ignore'           # Ignore extra environment variables not defined in Settings
    )
```
