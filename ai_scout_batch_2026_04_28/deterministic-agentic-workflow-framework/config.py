```python
import os
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized configuration settings for the Deterministic Agentic Workflow Framework.
    Settings are loaded from environment variables, with defaults provided.
    A .env file can be used to set environment variables locally.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra environment variables not defined here
        case_sensitive=False, # Allow flexible naming for env vars
    )

    # --- General Framework Settings ---
    APP_NAME: str = Field("Deterministic Agentic Workflow Framework", description="Name of the application")
    DEBUG: bool = Field(False, description="Enable debug mode for more verbose logging and potentially less optimized operations")
    LOG_LEVEL: str = Field("INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    LOG_FILE_PATH: str = Field("framework.log", description="Path to the log file")
    MAX_RETRIES: int = Field(3, description="Maximum number of retries for agent actions or LLM calls before failing")
    RETRY_BACKOFF_FACTOR: float = Field(0.5, description="Exponential backoff factor for retries (delay = factor * (2 ** (retry_num - 1)))")
    AGENT_TIMEOUT_SECONDS: int = Field(300, description="Default timeout for individual agent execution in seconds")

    # --- LLM Connector Settings ---
    # API keys are sensitive and should be loaded from environment variables
    OPENAI_API_KEY: SecretStr = Field(..., description="OpenAI API Key (required)")
    OPENAI_MODEL_NAME: str = Field("gpt-4o", description="Default OpenAI model to use for general tasks")
    OPENAI_COMPARATOR_MODEL_NAME: str = Field("gpt-3.5-turbo", description="OpenAI model to use for LLM-as-Comparator strategy")
    OPENAI_DEFAULT_TEMPERATURE: float = Field(0.7, description="Default temperature for OpenAI LLM calls")
    OPENAI_DEFAULT_MAX_TOKENS: int = Field(2048, description="Default max tokens for OpenAI LLM responses")
    OPENAI_EMBEDDING_MODEL_NAME: str = Field("text-embedding-3-small", description="OpenAI model for generating text embeddings")

    ANTHROPIC_API_KEY: SecretStr | None = Field(None, description="Anthropic API Key (optional)")
    ANTHROPIC_MODEL_NAME: str = Field("claude-3-opus-20240229", description="Default Anthropic model to use")
    ANTHROPIC_DEFAULT_TEMPERATURE: float = Field(0.7, description="Default temperature for Anthropic LLM calls")
    ANTHROPIC_DEFAULT_MAX_TOKENS: int = Field(2048, description="Default max tokens for Anthropic LLM responses")
    
    # --- Semantic Comparison Settings ---
    SEMANTIC_SIMILARITY_THRESHOLD: float = Field(0.8, ge=0.0, le=1.0, description="Cosine similarity threshold (0.0-1.0) for considering outputs semantically equivalent")
    SEMANTIC_DIFFERENCE_THRESHOLD: float = Field(0.2, ge=0.0, le=1.0, description="Threshold below which outputs are considered semantically divergent (1 - similarity_threshold)")
    
    # --- State Management Settings ---
    STATE_STORAGE_BACKEND: str = Field("sqlite", description="Backend for state storage ('in_memory', 'sqlite', 'postgres')")
    SQLITE_DATABASE_PATH: str = Field("workflow_state.db", description="Path for SQLite database file if 'sqlite' backend is used")
    POSTGRES_DATABASE_URL: str | None = Field(None, description="Connection URL for PostgreSQL database if 'postgres' backend is used (e.g., postgresql://user:password@host:port/database)")
    CHECKPOINT_INTERVAL_ACTIONS: int = Field(5, gt=0, description="Number of agent actions after which to automatically checkpoint the system state")
    CHECKPOINT_ENABLE_AUTOMATIC: bool = Field(True, description="Enable or disable automatic checkpointing after a set number of actions")
    STATE_SNAPSHOT_DIR: str = Field("state_snapshots", description="Directory to store state snapshots for forensic analysis or manual recovery")

    # --- Reconciliation Settings ---
    RECONCILIATION_ATTEMPTS: int = Field(2, description="Number of times the reconciliation agent will try to resolve an inconsistency")
    RECONCILIATION_MODEL_NAME: str = Field("gpt-4o", description="LLM model used by the reconciliation agent for analysis and proposing actions")

# Instantiate settings to be imported by other modules throughout the framework
# This creates a singleton configuration object that can be accessed globally.
settings = Settings()
```