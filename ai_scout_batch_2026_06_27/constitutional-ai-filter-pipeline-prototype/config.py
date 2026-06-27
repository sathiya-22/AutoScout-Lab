```python
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Configuration settings for the Constitutional AI Filter Pipeline.
    Loads environment variables for sensitive data like API keys.
    """
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="GEMINI_", case_sensitive=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure API key is loaded from GEMINI_API_KEY environment variable
        self.api_key = os.getenv("GEMINI_API_KEY", self.api_key)

```
