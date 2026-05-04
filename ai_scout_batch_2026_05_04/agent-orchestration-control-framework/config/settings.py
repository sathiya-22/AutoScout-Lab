import os

class Settings:
    """
    Global settings for the Agent Orchestration Control Framework.
    Settings can be overridden by environment variables where specified.
    """

    # --- General Application Settings ---
    APP_NAME: str = "Agent Orchestration Control Framework"
    APP_VERSION: str = "0.1.0"
    # Environment (e.g., "development", "testing", "production")
    ENVIRONMENT: str = os.getenv("APP_ENV", "development").lower()

    # --- Core Paths ---
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_DIR: str = os.path.join(BASE_DIR, "config")
    LOG_DIR: str = os.path.join(BASE_DIR, "logs")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    SPECIFICATIONS_DIR: str = os.path.join(BASE_DIR, "specifications")
    AGENTS_DIR: str = os.path.join(BASE_DIR, "agents")
    PROTOCOLS_DIR: str = os.path.join(BASE_DIR, "protocols")
    SUPERVISOR_DIR: str = os.path.join(BASE_DIR, "supervisor")
    MONITORING_DEBUGGING_DIR: str = os.path.join(BASE_DIR, "monitoring_debugging")

    # Ensure necessary directories exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Logging Settings ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper() # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FILE_PATH: str = os.path.join(LOG_DIR, "orchestrator.log")
    LOG_FORMAT: str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    LOG_TO_CONSOLE: bool = True # Whether to also log to standard output

    # --- State Manager Settings ---
    STATE_PERSISTENCE_ENABLED: bool = True
    STATE_PERSISTENCE_FILE: str = os.path.join(DATA_DIR, "orchestration_state.json")
    STATE_SAVE_INTERVAL_SECONDS: int = 10 # How often to save state to disk

    # --- Event Bus Settings ---
    # For a prototype, an in-memory bus is sufficient. Could be "redis", "kafka", etc.
    EVENT_BUS_TYPE: str = os.getenv("EVENT_BUS_TYPE", "in_memory")
    # If using Redis:
    # REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    # REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    # REDIS_DB: int = int(os.getenv("REDIS_DB", 0))

    # --- Agent Configuration Settings ---
    DEFAULT_AGENT_CONFIG_FILE: str = os.path.join(CONFIG_DIR, "agent_configs.yaml")
    AGENT_DEFAULT_TIMEOUT_SECONDS: int = 60 # Default timeout for an agent's single action

    # --- Tool Registry Settings ---
    TOOL_DEFINITIONS_PATH: str = os.path.join(AGENTS_DIR, "tool_definitions.yaml")

    # --- Formal Specification Settings (DSL) ---
    DSL_GRAMMAR_FILE: str = os.path.join(SPECIFICATIONS_DIR, "orchestration.grm") # Path to the DSL grammar definition
    DEFAULT_DSL_SPEC_FILE: str = os.path.join(SPECIFICATIONS_DIR, "example_spec_finance.fsl")
    CONSTRAINT_CHECK_INTERVAL_SECONDS: float = 0.5 # Frequency of constraint evaluation

    # --- Communication Protocol Settings ---
    MESSAGE_SCHEMA_PATH: str = os.path.join(PROTOCOLS_DIR, "message_schemas.json") # JSON schema for messages

    # --- Monitoring & Debugging Settings ---
    TRACER_OUTPUT_PATH: str = os.path.join(LOG_DIR, "execution_trace.jsonl") # Appending .jsonl for line-delimited JSON
    DEBUGGER_ENABLED: bool = os.getenv("DEBUGGER_ENABLED", "False").lower() == "true"
    VISUALIZER_ENABLED: bool = os.getenv("VISUALIZER_ENABLED", "False").lower() == "true"
    VISUALIZER_OUTPUT_PATH: str = os.path.join(DATA_DIR, "orchestration_graph.html") # HTML output for graph visualization
    METRICS_COLLECTION_ENABLED: bool = True

    # --- Supervisor/Control Plane Settings ---
    POLICY_ENGINE_RULES_FILE: str = os.path.join(SUPERVISOR_DIR, "policies.yaml")
    CONTROL_PLANE_API_ENABLED: bool = os.getenv("CONTROL_PLANE_API_ENABLED", "False").lower() == "true"
    CONTROL_PLANE_API_HOST: str = os.getenv("CONTROL_PLANE_API_HOST", "127.0.0.1")
    CONTROL_PLANE_API_PORT: int = int(os.getenv("CONTROL_PLANE_API_PORT", 8000))
    # Token or API Key for accessing the control plane
    CONTROL_PLANE_API_KEY: str = os.getenv("CONTROL_PLANE_API_KEY", "super-secret-key-123")


# Instantiate the settings for easy import
settings = Settings()