```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    """
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.3
    max_tokens: int = 500
    gemini_api_key: str = Field(validation_alias="GEMINI_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra='ignore' # Ignore extra fields not defined in the model
    )
```
