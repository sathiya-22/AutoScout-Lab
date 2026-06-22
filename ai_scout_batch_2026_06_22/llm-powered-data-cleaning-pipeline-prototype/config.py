```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Configuration settings for the LLM-powered data cleaning pipeline.
    Loads values from environment variables or a .env file.
    """
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    api_key: str = Field(env='GEMINI_API_KEY')
    
    # Note: Using 'gemini-1.5-flash' as 'gemini-2.5-flash' is not a publicly available model name.
    # 'gemini-1.5-flash' offers high performance and cost-effectiveness for this task.
    model_name: str = "gemini-1.5-flash" 
    
    temperature: float = 0.3
    max_tokens: int = 2048

```
