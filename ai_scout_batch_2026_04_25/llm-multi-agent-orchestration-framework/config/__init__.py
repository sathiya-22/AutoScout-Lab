```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional

class Settings(BaseSettings):
    """
    Centralized configuration for the multi-agent orchestration framework.
    Settings can be loaded from environment variables or a .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=False,  # Allow case-insensitive env var names
        extra='ignore'         # Ignore environment variables not explicitly defined
    )

    # --- General System Settings ---
    ENVIRONMENT: Literal["development", "production", "testing"] = "development"
    DEBUG: bool = False
    SYSTEM_NAME: str = "MAOF_Orchestrator" # Multi-Agent Orchestration Framework
    AGENT_HEARTBEAT_INTERVAL_SECONDS: int = 15 # How often agents report their status to the orchestrator

    # --- LLM Settings ---
    DEFAULT_LLM_MODEL: str = "gpt-4o" # e.g., gpt-4o, claude-3-opus-20240229, llama3
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    LLM_DEFAULT_TEMPERATURE: float = 0.7
    LLM_DEFAULT_MAX_TOKENS: int = 4096 # Max tokens for LLM responses
    LLM_TIMEOUT_SECONDS: int = 120 # Timeout for LLM API calls

    # --- Communication Primitives Settings ---
    # Message Queues (e.g., Redis Streams, Kafka)
    MESSAGE_QUEUE_TYPE: Literal["redis", "kafka", "in-memory"] = "redis"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0 # Default database for general use or message queues
    REDIS_PASSWORD: Optional[str] = None
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092" # Comma-separated list of brokers

    # Shared Memory/Stigmergy (e.g., Redis)
    STIGMERGY_STORE_TYPE: Literal["redis", "in-memory"] = "redis"
    # Stigmergy can use separate Redis settings or reuse general ones
    STIGMERGY_REDIS_DB: int = 1 # Use a different DB for stigmergy to avoid conflicts

    # RPC Interface (for synchronous direct calls)
    RPC_HOST: str = "0.0.0.0" # Default host for RPC servers to bind to
    RPC_PORT_RANGE_START: int = 8000 # Start of port range for dynamic RPC agent ports
    RPC_PORT_RANGE_END: int = 9000   # End of port range

    # --- Resource Management Settings ---
    MAX_SYSTEM_LLM_TOKENS_PER_HOUR: int = 5_000_000 # System-wide LLM token usage limit
    MAX_SYSTEM_LLM_COST_PER_DAY: float = 500.0   # System-wide LLM cost limit in USD
    DEFAULT_AGENT_LLM_TOKEN_ALLOCATION: int = 100_000 # Default tokens per agent per hour
    DEFAULT_AGENT_COMPUTE_ALLOCATION_UNITS: float = 0.5 # e.g., virtual CPU cores
    DEFAULT_AGENT_MEMORY_ALLOCATION_MB: int = 256 # Default memory in MB

    # --- Monitoring & Observability Settings ---
    LOG_LEVEL: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"] = "INFO"
    METRICS_ENABLED: bool = True
    TRACING_ENABLED: bool = True
    PROMETHEUS_PORT: int = 8001 # Port for Prometheus metrics exposition
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317" # OpenTelemetry collector GRPC endpoint

    # --- Task Management Settings ---
    TASK_QUEUE_MAX_SIZE: int = 1000 # Maximum number of tasks in the global orchestrator queue
    TASK_DEFAULT_PRIORITY: int = 5 # Default priority for new tasks (1 highest, 10 lowest)
    TASK_RETRY_ATTEMPTS: int = 3 # Number of times to retry a failed task before marking as permanently failed

    # --- Error Handling Settings ---
    DEFAULT_RETRY_DELAY_SECONDS: int = 5 # Default delay between retries
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5 # Consecutive failures before opening circuit
    CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS: int = 300 # Time (seconds) before attempting to close circuit

    # --- Orchestrator Specific Settings ---
    ORCHESTRATOR_ID: str = "main-orchestrator-001" # Unique ID for the orchestrator instance

# Instantiate the settings object to be imported
settings = Settings()

# --- Basic Post-Initialization Validation and Error Handling ---
if settings.ENVIRONMENT == "production" and settings.DEBUG:
    print("WARNING: Debug mode is enabled in a production environment. This is not recommended.")

if settings.MESSAGE_QUEUE_TYPE == "kafka" and not settings.KAFKA_BOOTSTRAP_SERVERS:
    raise ValueError(
        "KAFKA_BOOTSTRAP_SERVERS must be specified when MESSAGE_QUEUE_TYPE is 'kafka'."
    )

if settings.MESSAGE_QUEUE_TYPE == "redis" and (not settings.REDIS_HOST or not settings.REDIS_PORT):
    raise ValueError(
        "REDIS_HOST and REDIS_PORT must be specified when MESSAGE_QUEUE_TYPE is 'redis'."
    )

# Check for LLM API keys based on the default model in production
if settings.ENVIRONMENT == "production":
    if settings.DEFAULT_LLM_MODEL.startswith("gpt") and not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be set for OpenAI models in production.")
    if settings.DEFAULT_LLM_MODEL.startswith("claude") and not settings.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY must be set for Anthropic models in production.")
    if settings.DEFAULT_LLM_MODEL.startswith("gemini") and not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY must be set for Google models in production.")
```