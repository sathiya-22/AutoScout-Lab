```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Configuration settings for the Task Decomposition and Planning Agent.
    Reads environment variables, optionally from a .env file.
    """
    model_name: str = "gemini-2.5-flash"  # The generative model to use
    temperature: float = 0.3             # Controls creativity (0.0-1.0)
    max_tokens: int = 2048               # Maximum number of tokens in the output
    api_key: str                         # Google Gemini API key

    model_config = SettingsConfigDict(
        env_file='.env',
        extra='ignore',
        env_prefix='GEMINI_', # Looks for environment variables prefixed with GEMINI_ (e.g., GEMINI_API_KEY)
    )
```
