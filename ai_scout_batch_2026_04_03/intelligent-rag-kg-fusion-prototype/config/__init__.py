import os

class BaseConfig:
    """Base configuration for the Intelligent RAG & Knowledge Graph Fusion System."""
    PROJECT_NAME: str = "Intelligent RAG & Knowledge Graph Fusion System"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    DATA_DIR: str = os.getenv("DATA_DIR", "data") # Directory for local persistent storage (e.g., Chroma DB, SQLite DB, cached embeddings)
    APP_ENV: str = os.getenv("APP_ENV", "development") # Options: development, staging, production

class LLMConfig:
    """Configuration for Large Language Model Providers."""
    DEFAULT_MODEL_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai") # e.g., "openai", "anthropic", "google", "groq", "huggingface"
    DEFAULT_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o") # Example: "gpt-4o", "claude-3-opus-20240229", "llama3-8b-8192"

    # OpenAI Specific
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") # Recommended: Set as environment variable
    OPENAI_MODEL_TEMPERATURE: float = float(os.getenv("OPENAI_MODEL_TEMPERATURE", 0.3))
    OPENAI_MODEL_MAX_TOKENS: int = int(os.getenv("OPENAI_MODEL_MAX_TOKENS", 4096))

    # Anthropic Specific
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL_TEMPERATURE: float = float(os.getenv("ANTHROPIC_MODEL_TEMPERATURE", 0.3))
    ANTHROPIC_MODEL_MAX_TOKENS: int = int(os.getenv("ANTHROPIC_MODEL_MAX_TOKENS", 4096))

    # Google Specific
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    GOOGLE_MODEL_TEMPERATURE: float = float(os.getenv("GOOGLE_MODEL_TEMPERATURE", 0.3))
    GOOGLE_MODEL_MAX_TOKENS: int = int(os.getenv("GOOGLE_MODEL_MAX_TOKENS", 4096))

    # Groq Specific
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    GROQ_MODEL_TEMPERATURE: float = float(os.getenv("GROQ_MODEL_TEMPERATURE", 0.3))
    GROQ_MODEL_MAX_TOKENS: int = int(os.getenv("GROQ_MODEL_MAX_TOKENS", 4096))
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192") # Specific Groq model if different from general default

    # Fallback/General LLM settings if provider-specific ones are not set
    # These will be used if the specific provider config isn't available or relevant
    MODEL_TEMPERATURE: float = float(os.getenv("LLM_MODEL_TEMPERATURE", 0.3))
    MODEL_MAX_TOKENS: int = int(os.getenv("LLM_MODEL_MAX_TOKENS", 4096))

class EmbeddingConfig:
    """Configuration for embedding models."""
    MODEL_PROVIDER: str = os.getenv("EMBEDDING_MODEL_PROVIDER", "openai") # e.g., "openai", "huggingface", "cohere"
    MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small") # Example: "text-embedding-ada-002", "BAAI/bge-small-en-v1.5", "intfloat/multilingual-e5-large"
    DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", 1536)) # Output dimensions of the embedding model. Ensure this matches the model.
    HUGGINGFACE_MODEL_PATH: str = os.getenv("HUGGINGFACE_EMBEDDING_MODEL_PATH", None) # Path to local Hugging Face model if provider is 'huggingface' and not using a remote API

class VectorDBConfig:
    """Configuration for the Vector Database."""
    PROVIDER: str = os.getenv("VECTOR_DB_PROVIDER", "chroma") # Options: "chroma", "qdrant", "pinecone", "weaviate"

    # ChromaDB Specific (local persistent)
    CHROMA_PATH: str = os.path.join(BaseConfig.DATA_DIR, os.getenv("CHROMA_DB_NAME", "chroma_db"))

    # Qdrant Specific
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", 6334))
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", None) # For cloud or authenticated instances

    # Pinecone Specific
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", None)
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", None)
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rag-index")

    # Weaviate Specific
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", None)
    WEAVIATE_API_KEY: str = os.getenv("WEAVIATE_API_KEY", None)

class KnowledgeGraphDBConfig:
    """Configuration for the Knowledge Graph Database."""
    PROVIDER: str = os.getenv("KG_DB_PROVIDER", "neo4j") # Options: "neo4j", "rdf_store", "gremlin", "custom_in_memory"

    # Neo4j Specific
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Gremlin Specific (e.g., AWS Neptune)
    GREMLIN_ENDPOINT: str = os.getenv("GREMLIN_ENDPOINT", None)

class StructuredDBConfig:
    """Configuration for Structured Databases."""
    PROVIDER: str = os.getenv("STRUCTURED_DB_PROVIDER", "sqlite") # Options: "sqlite", "postgresql", "mysql", "mongodb"

    # SQLite Specific (local file)
    SQLITE_PATH: str = os.path.join(BaseConfig.DATA_DIR, os.getenv("SQLITE_DB_NAME", "structured_data.db"))

    # PostgreSQL Specific
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "user")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "rag_structured_db")

    # MongoDB Specific
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "rag_mongodb")

class IngestionConfig:
    """Configuration for data ingestion pipeline."""
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("INGESTION_CHUNK_SIZE", 1000)) # Max tokens per chunk
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("INGESTION_CHUNK_OVERLAP", 100)) # Overlap between chunks for context
    MIN_CHUNK_SIZE: int = int(os.getenv("INGESTION_MIN_CHUNK_SIZE", 50)) # Minimum tokens per chunk

    # Strategy for logical chunking (e.g., "recursive", "semantic", "by_section")
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "recursive")
    
    # LLM for Knowledge Graph triple extraction during ingestion
    KG_EXTRACTION_MODEL: str = os.getenv("KG_EXTRACTION_MODEL", LLMConfig.DEFAULT_MODEL_NAME)
    KG_EXTRACTION_MODEL_TEMP: float = float(os.getenv("KG_EXTRACTION_MODEL_TEMP", 0.1))

    # Metadata extraction settings
    PRESERVE_METADATA_ON_CHUNK: bool = True # Whether to carry source metadata to each chunk
    AUTHORITY_SCORE_METHOD: str = os.getenv("AUTHORITY_SCORE_METHOD", "heuristic") # "heuristic", "ml_model", "manual"
    FRESHNESS_DATE_FIELD: str = "last_updated" # The metadata field that stores freshness date

class RetrievalConfig:
    """Configuration for multi-modal/multi-strategy retrieval."""
    TOP_K_VECTOR_SEARCH: int = int(os.getenv("RETRIEVAL_TOP_K_VECTOR", 5)) # Number of results from vector store
    TOP_K_KG_SEARCH: int = int(os.getenv("RETRIEVAL_TOP_K_KG", 3)) # Number of paths/entities from KG
    TOP_K_STRUCTURED_SEARCH: int = int(os.getenv("RETRIEVAL_TOP_K_STRUCTURED", 2)) # Number of rows/results from structured DB

    # Weights for combining different retrieval results in hybrid retriever (must sum to 1.0 if used for weighted sum)
    HYBRID_RETRIEVER_WEIGHTS: dict = {
        "vector": float(os.getenv("HYBRID_WEIGHT_VECTOR", 0.6)),
        "kg": float(os.getenv("HYBRID_WEIGHT_KG", 0.3)),
        "structured": float(os.getenv("HYBRID_WEIGHT_STRUCTURED", 0.1))
    }
    HYBRID_RETRIEVAL_METHOD: str = os.getenv("HYBRID_RETRIEVAL_METHOD", "rerank") # Options: "rerank", "weighted_sum", "round_robin"

    # Metadata filtering and ranking thresholds
    MIN_AUTHORITY_SCORE: float = float(os.getenv("MIN_AUTHORITY_SCORE", 0.6)) # Minimum score (0.0-1.0) to consider a source highly authoritative
    MAX_FRESHNESS_DAYS: int = int(os.getenv("MAX_FRESHNESS_DAYS", 180)) # Max age in days for a source to be considered fresh for priority ranking
    ENABLE_KEYWORD_SEARCH: bool = True # Flag to enable/disable basic keyword search (can be integrated or separate)

    # Re-ranking model configuration (optional)
    RERANKER_ENABLED: bool = bool(os.getenv("RERANKER_ENABLED", "False").lower() == "true")
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base") # Example: "cross-encoder/ms-marco-TinyBERT-L-2"
    RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", 10)) # Number of documents to pass to reranker

class AgentConfig:
    """Configuration for agentic query refinement."""
    MAX_REFINEMENT_ITERATIONS: int = int(os.getenv("AGENT_MAX_REFINEMENT_ITERATIONS", 3))
    AGENT_MODEL_NAME: str = os.getenv("AGENT_MODEL_NAME", LLMConfig.DEFAULT_MODEL_NAME) # LLM specifically for the agent if different from default
    AGENT_MODEL_TEMPERATURE: float = float(os.getenv("AGENT_MODEL_TEMPERATURE", 0.2)) # A bit lower for more deterministic agent behavior
    AGENT_TOOL_TIMEOUT: int = int(os.getenv("AGENT_TOOL_TIMEOUT", 60)) # Seconds before a tool call times out

    # Prompt engineering specific settings for agents
    AGENT_SYSTEM_PROMPT_PATH: str = os.path.join("src", "agents", "prompts", "system_prompt.txt")
    AGENT_TOOLS_DESCRIPTION_PATH: str = os.path.join("src", "agents", "prompts", "tools_description.txt")

class GenerationConfig:
    """Configuration for response generation."""
    LLM_MODEL_NAME_GENERATION: str = os.getenv("LLM_MODEL_NAME_GENERATION", LLMConfig.DEFAULT_MODEL_NAME) # LLM for final response synthesis
    LLM_TEMPERATURE_GENERATION: float = float(os.getenv("LLM_TEMPERATURE_GENERATION", 0.7)) # Higher temperature for more creative/diverse answers
    LLM_MAX_TOKENS_GENERATION: int = int(os.getenv("LLM_MAX_TOKENS_GENERATION", 2048)) # Max tokens for the generated response
    
    CITATION_STYLE: str = os.getenv("CITATION_STYLE", "numbered_links") # Options: "numbered_links", "apa_style", "inline_text"
    INCLUDE_SOURCE_METADATA_IN_RESPONSE: bool = True # Whether to include source metadata details in the final response

class Settings:
    """Aggregates all configuration settings for easy access."""
    BASE = BaseConfig()
    LLM = LLMConfig()
    EMBEDDING = EmbeddingConfig()
    VECTOR_DB = VectorDBConfig()
    KNOWLEDGE_GRAPH_DB = KnowledgeGraphDBConfig()
    STRUCTURED_DB = StructuredDBConfig()
    INGESTION = IngestionConfig()
    RETRIEVAL = RetrievalConfig()
    AGENT = AgentConfig()
    GENERATION = GenerationConfig()

# Instantiate settings for direct import and access throughout the application
settings = Settings()

# Example of basic error handling/validation:
# Ensure critical API keys are present in production environment
if settings.BASE.APP_ENV == "production":
    if settings.LLM.DEFAULT_MODEL_PROVIDER == "openai" and not settings.LLM.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be set in production environment.")
    # Add similar checks for other critical services