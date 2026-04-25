import os
from typing import List, Literal, Optional, Dict, Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    """
    Centralized configuration management for the multi-agent orchestration framework.
    Settings are loaded from environment variables and can be overridden.
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- General System Settings ---
    ENV: Literal["development", "production", "testing"] = Field(
        "development", description="The current environment (development, production, testing)."
    )
    SERVICE_NAME: str = Field(
        "MultiAgentOrchestrator", description="Name of the orchestration service."
    )
    SYSTEM_ID: str = Field(
        "default-orchestration-system", description="Unique identifier for this orchestration system instance."
    )
    
    # --- Orchestrator Settings ---
    ORCHESTRATOR_ID: str = Field(
        "orchestrator-001", description="Unique identifier for the orchestrator instance."
    )
    AGENT_HEARTBEAT_INTERVAL_SECONDS: int = Field(
        30, description="Interval in seconds for agents to send heartbeat signals to the orchestrator."
    )
    TASK_SCHEDULING_STRATEGY: Literal["round_robin", "least_loaded", "capability_match", "priority"] = Field(
        "capability_match", description="Strategy for scheduling tasks to agents (e.g., capability_match, least_loaded)."
    )
    
    # --- Agent Settings ---
    DEFAULT_LLM_MODEL: str = Field(
        "gpt-4o", description="Default LLM model to be used by agents if not specified."
    )
    OPENAI_API_KEY: Optional[SecretStr] = Field(
        None, description="API key for OpenAI models.", alias="OPENAI_API_KEY"
    )
    ANTHROPIC_API_KEY: Optional[SecretStr] = Field(
        None, description="API key for Anthropic models.", alias="ANTHROPIC_API_KEY"
    )
    GOOGLE_API_KEY: Optional[SecretStr] = Field(
        None, description="API key for Google models.", alias="GOOGLE_API_KEY"
    )
    LLM_API_TIMEOUT_SECONDS: int = Field(
        120, description="Timeout for LLM API calls in seconds."
    )
    DEFAULT_AGENT_CAPACITY_TOKENS_PER_MINUTE: int = Field(
        150000, description="Default token capacity limit for agents per minute."
    )
    DEFAULT_AGENT_CONCURRENT_TASKS: int = Field(
        5, description="Default maximum number of concurrent tasks an agent can handle."
    )
    AGENT_CONFIG_DIR: str = Field(
        "config/agents", description="Directory where agent-specific configuration files are stored for the AgentFactory."
    )

    # --- Communication Settings ---
    # Message Queues (e.g., Redis Streams, Kafka)
    MESSAGE_QUEUE_TYPE: Literal["REDIS", "KAFKA", "IN_MEMORY"] = Field(
        "REDIS", description="Type of message queue backend to use (REDIS, KAFKA, IN_MEMORY for testing)."
    )
    REDIS_HOST: str = Field(
        "localhost", description="Redis server host for message queues and shared memory."
    )
    REDIS_PORT: int = Field(
        6379, description="Redis server port."
    )
    REDIS_DB_MESSAGES: int = Field(
        0, description="Redis database index for message queues."
    )
    REDIS_DB_SHARED_MEMORY: int = Field(
        1, description="Redis database index for shared memory (stigmergy)."
    )
    KAFKA_BROKERS: List[str] = Field(
        ["localhost:9092"], description="List of Kafka broker addresses (e.g., ['localhost:9092'])."
    )
    KAFKA_TOPIC_AGENT_MAIL: str = Field(
        "agent_mail", description="Kafka topic for direct agent communication."
    )
    KAFKA_TOPIC_SYSTEM_EVENTS: str = Field(
        "system_events", description="Kafka topic for system-wide events and broadcasts."
    )
    
    # Shared Memory/Stigmergy (e.g., Redis, Distributed KV Store)
    SHARED_MEMORY_TYPE: Literal["REDIS", "IN_MEMORY"] = Field(
        "REDIS", description="Type of shared memory backend to use for stigmergy (REDIS, IN_MEMORY for testing)."
    )
    SHARED_MEMORY_ENTRY_TTL_SECONDS: int = Field(
        3600, description="Default TTL for shared memory entries in seconds (1 hour)."
    )
    
    # RPC Interface (e.g., FastAPI/HTTP, gRPC)
    RPC_PROTOCOL: Literal["HTTP", "GRPC"] = Field(
        "HTTP", description="Protocol for RPC communication (HTTP or GRPC)."
    )
    RPC_HOST: str = Field(
        "0.0.0.0", description="Default host address for RPC servers (agents or orchestrator)."
    )
    RPC_PORT: int = Field(
        8000, description="Default port for RPC servers."
    )
    
    # --- Resource Management Settings ---
    RESOURCE_MONITOR_INTERVAL_SECONDS: int = Field(
        10, description="Interval in seconds to monitor agent resource usage (e.g., tokens, compute)."
    )
    MAX_TOTAL_TOKEN_USAGE_DAILY: Optional[int] = Field(
        None, description="Maximum total token usage across all agents per day (for cost control). None for no limit."
    )
    
    # --- Monitoring & Observability Settings ---
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", description="Minimum logging level for the system."
    )
    LOG_FORMAT: Literal["JSON", "TEXT"] = Field(
        "TEXT", description="Format for log output (JSON or TEXT)."
    )
    PROMETHEUS_EXPORTER_PORT: Optional[int] = Field(
        8001, description="Port for the Prometheus metrics exporter. Set to None to disable."
    )
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = Field(
        None, description="OpenTelemetry collector endpoint for tracing (e.g., 'http://localhost:4317'). Set to None to disable."
    )
    TRACING_ENABLED: bool = Field(
        False, description="Enable or disable distributed tracing."
    )

    # --- Error Handling Settings ---
    DEFAULT_RETRY_ATTEMPTS: int = Field(
        3, description="Default number of retry attempts for failed operations."
    )
    DEFAULT_RETRY_DELAY_SECONDS: int = Field(
        5, description="Default delay in seconds between retry attempts."
    )
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        5, description="Number of consecutive failures before a circuit breaker trips (opens)."
    )
    CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS: int = Field(
        60, description="Time in seconds before a tripped circuit breaker attempts to reset (half-open state)."
    )

    # --- Development/Testing Specific Settings ---
    DEVELOPMENT_MODE_ENABLED: bool = Field(
        False, description="If true, enables development-specific features like hot-reloading or verbose debugging."
    )
    MOCK_COMMUNICATION_ENABLED: bool = Field(
        False, description="If true, communication primitives will use in-memory mocks for testing."
    )

# Instantiate and export settings
settings = AppSettings()