import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """
    Centralized configuration for the Intelligent RAG & Knowledge Graph Fusion System.
    Sensitive information is loaded from environment variables for security.
    """

    # --- General Application Settings ---
    APP_NAME: str = "Intelligent RAG & KG Fusion System"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development") # 'development', 'production', 'testing'
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO") # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    # Base directory for data storage, logs, etc.
    BASE_DATA_DIR: str = os.getenv("BASE_DATA_DIR", "data/")

    # --- LLM Settings (for Generation and Agentic Query Refinement) ---
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai") # e.g., 'openai', 'anthropic', 'huggingface'
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o") # Specific model identifier
    LLM_API_KEY: str = os.getenv("LLM_API_KEY") # Required for external LLM providers
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 1024))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", 120)) # Seconds

    # --- Embedding Model Settings (for Vector Store) ---
    EMBEDDING_MODEL_PROVIDER: str = os.getenv("EMBEDDING_MODEL_PROVIDER", "openai") # e.g., 'openai', 'huggingface', 'sentence-transformers'
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small") # Specific model identifier
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY) # Often same as LLM API key
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", 1536)) # Output dimension of the embedding model

    # --- Vector Store Settings (for Semantic Search) ---
    VECTOR_STORE_PROVIDER: str = os.getenv("VECTOR_STORE_PROVIDER", "chroma") # e.g., 'chroma', 'pinecone', 'weaviate'
    VECTOR_STORE_HOST: str = os.getenv("VECTOR_STORE_HOST", "localhost")
    VECTOR_STORE_PORT: int = int(os.getenv("VECTOR_STORE_PORT", 8000))
    VECTOR_STORE_API_KEY: str = os.getenv("VECTOR_STORE_API_KEY", "") # If cloud service or secured
    VECTOR_STORE_INDEX_NAME: str = os.getenv("VECTOR_STORE_INDEX_NAME", "rag_chunks")
    VECTOR_STORE_COLLECTION_NAME: str = os.getenv("VECTOR_STORE_COLLECTION_NAME", "default_collection")
    # For Chroma, if using a persistent client
    CHROMA_PERSIST_DIRECTORY: str = os.path.join(BASE_DATA_DIR, "chroma_db")

    # --- Knowledge Graph Store Settings (for Relationship Mapping and Factual Lookup) ---
    KG_STORE_PROVIDER: str = os.getenv("KG_STORE_PROVIDER", "neo4j") # e.g., 'neo4j', 'gdb', 'rdf_store'
    KG_STORE_URI: str = os.getenv("KG_STORE_URI", "bolt://localhost:7687")
    KG_STORE_USER: str = os.getenv("KG_STORE_USER", "neo4j")
    KG_STORE_PASSWORD: str = os.getenv("KG_STORE_PASSWORD") # Required for KG DB
    KG_STORE_DATABASE_NAME: str = os.getenv("KG_STORE_DATABASE_NAME", "neo4j")
    # For KG embedding search, if applicable
    KG_EMBEDDING_MODEL_NAME: str = os.getenv("KG_EMBEDDING_MODEL_NAME", EMBEDDING_MODEL_NAME)

    # --- Structured Data Store Settings (for Numerical Computation and Specific Data Points) ---
    STRUCTURED_DB_PROVIDER: str = os.getenv("STRUCTURED_DB_PROVIDER", "sqlite") # e.g., 'postgresql', 'mongodb', 'sqlite'
    # Connection string for the structured database
    STRUCTURED_DB_URI: str = os.getenv("STRUCTURED_DB_URI", f"sqlite:///{BASE_DATA_DIR}/structured_data.db")
    # Optional, specific database name if not in URI
    STRUCTURED_DB_NAME: str = os.getenv("STRUCTURED_DB_NAME", "rag_structured_db")

    # --- Ingestion Pipeline Settings ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512)) # Max tokens per chunk
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50)) # Tokens overlap between chunks
    MIN_CHUNK_LENGTH: int = int(os.getenv("MIN_CHUNK_LENGTH", 20)) # Minimum tokens for a valid chunk
    DOCUMENT_LOADER_TIMEOUT: int = int(os.getenv("DOCUMENT_LOADER_TIMEOUT", 30)) # Seconds for loading
    # List of enabled metadata extractors (e.g., 'author', 'publication_date', 'data_type')
    METADATA_EXTRACTORS_ENABLED: list[str] = os.getenv("METADATA_EXTRACTORS_ENABLED", "source_authority,freshness,data_type").split(',')
    # Configuration for KG entity/relationship extraction
    KG_ENTITY_TYPES: list[str] = os.getenv("KG_ENTITY_TYPES", "PERSON,ORGANIZATION,LOCATION,EVENT,PRODUCT,DATE,NUMBER").split(',')
    KG_RELATIONSHIP_TYPES: list[str] = os.getenv("KG_RELATIONSHIP_TYPES", "WORKS_FOR,LOCATED_IN,PART_OF,HAS_PROPERTY,MENTIONS").split(',')

    # --- Retrieval & Ranking Settings ---
    # Weights for hybrid retrieval fusion or re-ranking
    VECTOR_SCORE_WEIGHT: float = float(os.getenv("VECTOR_SCORE_WEIGHT", 0.6))
    KG_SCORE_WEIGHT: float = float(os.getenv("KG_SCORE_WEIGHT", 0.3))
    STRUCTURED_SCORE_WEIGHT: float = float(os.getenv("STRUCTURED_SCORE_WEIGHT", 0.1))
    # Weights for metadata filtering/ranking
    AUTHORITY_SCORE_WEIGHT: float = float(os.getenv("AUTHORITY_SCORE_WEIGHT", 0.4))
    FRESHNESS_SCORE_WEIGHT: float = float(os.getenv("FRESHNESS_SCORE_WEIGHT", 0.3))
    RELEVANCE_SCORE_WEIGHT: float = float(os.getenv("RELEVANCE_SCORE_WEIGHT", 0.3))
    # Max days a document is considered 'fresh' for scoring
    MAX_FRESHNESS_DAYS: int = int(os.getenv("MAX_FRESHNESS_DAYS", 365)) # 1 year

    # --- Agentic Query Refinement Settings ---
    # The LLM model to use specifically for agents (can be different from generation LLM)
    AGENT_LLM_MODEL_NAME: str = os.getenv("AGENT_LLM_MODEL_NAME", "gpt-4o-mini")
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", 3)) # Max refinement steps
    AGENT_TOOL_TIMEOUT: int = int(os.getenv("AGENT_TOOL_TIMEOUT", 60)) # Seconds for an agent tool call

    # --- Evaluation Settings ---
    EVALS_ENABLED: bool = os.getenv("EVALS_ENABLED", "False").lower() == "true"
    EVAL_DATASET_PATH: str = os.path.join(BASE_DATA_DIR, "evals/evaluation_dataset.json")
    EVAL_OUTPUT_PATH: str = os.path.join(BASE_DATA_DIR, "evals/results/")

    # --- Error Handling & Fallbacks ---
    RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", 3))
    RETRY_DELAY_SECONDS: int = int(os.getenv("RETRY_DELAY_SECONDS", 5))

    def __init__(self):
        # Basic validation for critical API keys
        if self.LLM_PROVIDER in ["openai", "anthropic"] and not self.LLM_API_KEY:
            raise ValueError(f"LLM_API_KEY is not set for provider: {self.LLM_PROVIDER}")
        if self.EMBEDDING_MODEL_PROVIDER in ["openai"] and not self.EMBEDDING_API_KEY:
            raise ValueError(f"EMBEDDING_API_KEY is not set for provider: {self.EMBEDDING_MODEL_PROVIDER}")
        if self.KG_STORE_PROVIDER == "neo4j" and not self.KG_STORE_PASSWORD:
            raise ValueError("KG_STORE_PASSWORD is not set for Neo4j provider.")

# Instantiate settings for easy import
settings = Settings()