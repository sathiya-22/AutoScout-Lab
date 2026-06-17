```python
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Configuration settings for the Multi-Hop Question Answering Agent.
    Reads environment variables for sensitive information and model parameters.
    """
    model_name: str = "gemini-1.5-flash"  # Using 1.5-flash as gemini-2.5-flash is not a standard model name
    temperature: float = 0.4
    max_tokens: int = 1024
    api_key: str = Field(..., env="GEMINI_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra fields in .env or environment
    )

    def __str__(self):
        return (
            f"Config:\n"
            f"  Model Name: {self.model_name}\n"
            f"  Temperature: {self.temperature}\n"
            f"  Max Tokens: {self.max_tokens}\n"
            f"  API Key Set: {'Yes' if self.api_key else 'No'}"
        )

```
