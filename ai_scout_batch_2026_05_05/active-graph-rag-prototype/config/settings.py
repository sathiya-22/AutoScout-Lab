from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    """
    Manages environment variables, API keys, and global configuration parameters
    for the Graph RAG prototype. Settings are loaded from environment variables
    and optionally from a .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" # Ignore extra environment variables not defined here
    )

    # --- General Application Settings ---
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development") # 'development', 'production', 'testing'
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper() # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    DATA_DIR: str = os.getenv("DATA_DIR", "data") # Base directory for data storage (e.g., processed documents)
    PROMPT_TEMPLATES_PATH: str = os.getenv("PROMPT_TEMPLATES_PATH", "config/prompt_templates.py") # Path to prompt templates file

    # --- Graph Database Settings ---
    GRAPH_DB_TYPE: str = os.getenv("GRAPH_DB_TYPE", "NEO4J").upper() # e.g., 'NEO4J', 'TINKERPOP', 'JANUSGRAPH'
    GRAPH_DB_HOST: str = os.getenv("GRAPH_DB_HOST", "localhost")
    GRAPH_DB_PORT: int = int(os.getenv("GRAPH_DB_PORT", "7687")) # Default Bolt port for Neo4j
    GRAPH_DB_USER: str = os.getenv("GRAPH_DB_USER", "neo4j")
    GRAPH_DB_PASSWORD: str = os.getenv("GRAPH_DB_PASSWORD", "password")
    GRAPH_DB_NAME: str = os.getenv("GRAPH_DB_NAME", "neo4j") # For multi-database setups (e.g., Neo4j)

    # --- LLM Settings (for extraction, generation, self-correction) ---
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower() # e.g., 'openai', 'anthropic', 'huggingface'
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "") # API key for the chosen LLM provider
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "") # For custom endpoints or local LLMs (e.g., vLLM, Ollama)
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1")) # Controls randomness
    LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))
    LLM_REQUEST_TIMEOUT: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "120")) # Timeout for LLM API calls in seconds

    # --- Embedding Service Settings ---
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai").lower() # e.g., 'openai', 'huggingface', 'cohere'
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "") # Can be same as LLM_API_KEY for OpenAI
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536")) # Output dimension of the embedding model
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100")) # Number of items to send per embedding API call

    # --- Document Processing Settings ---
    DOCUMENT_CHUNK_SIZE: int = int(os.getenv("DOCUMENT_CHUNK_SIZE", "1000")) # Max characters per text chunk
    DOCUMENT_CHUNK_OVERLAP: int = int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "100")) # Overlap between chunks
    SUPPORTED_DOCUMENT_TYPES: list[str] = os.getenv("SUPPORTED_DOCUMENT_TYPES", "pdf,html,txt,md").split(',') # File extensions

    # --- Active Learning Settings ---
    # Threshold for flagging extractions as uncertain (e.g., lower confidence scores)
    ACTIVE_LEARNING_UNCERTAINTY_THRESHOLD: float = float(os.getenv("ACTIVE_LEARNING_UNCERTAINTY_THRESHOLD", "0.2"))
    # Number of samples to present for human review in one batch
    ACTIVE_LEARNING_BATCH_SIZE: int = int(os.getenv("ACTIVE_LEARNING_BATCH_SIZE", "50"))
    # Strategy for selecting samples: 'confidence', 'entropy', 'graph_impact', 'random'
    ACTIVE_LEARNING_SELECTION_STRATEGY: str = os.getenv("ACTIVE_LEARNING_SELECTION_STRATEGY", "confidence").lower()

    # --- Self-Correction Settings ---
    # Maximum attempts for the LLM to self-correct a problematic extraction
    SELF_CORRECTION_MAX_RETRIES: int = int(os.getenv("SELF_CORRECTION_MAX_RETRIES", "3"))
    # Optional: Use a different, potentially more capable LLM for self-correction
    SELF_CORRECTION_LLM_MODEL_NAME: str = os.getenv("SELF_CORRECTION_LLM_MODEL_NAME", LLM_MODEL_NAME)

    # --- Graph Reconciliation Settings ---
    # Similarity threshold for entity resolution (e.g., embedding similarity, string similarity)
    ENTITY_RESOLUTION_SIMILARITY_THRESHOLD: float = float(os.getenv("ENTITY_RESOLUTION_SIMILARITY_THRESHOLD", "0.85"))
    # Strategy for resolving conflicts: 'new_over_old', 'old_over_new', 'human_review', 'llm_decide'
    CONFLICT_RESOLUTION_STRATEGY: str = os.getenv("CONFLICT_RESOLUTION_STRATEGY", "llm_decide").lower()

    # --- RAG Settings ---
    # Max tokens to include in the context passed to the RAG LLM
    RAG_MAX_CONTEXT_TOKENS: int = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "4000"))
    # Number of hops to traverse in the KG for context retrieval
    RAG_GRAPH_RETRIEVAL_HOPS: int = int(os.getenv("RAG_GRAPH_RETRIEVAL_HOPS", "2"))
    # Number of top relevant graph elements (nodes/relations) to retrieve
    RAG_RETRIEVER_TOP_K: int = int(os.getenv("RAG_RETRIEVER_TOP_K", "5"))


# Instantiate settings to be imported by other modules
settings = Settings()