import os

class Config:
    PROJECT_NAME = "Hybrid Semantic and Logical RAG"

    # --- General Settings ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO") # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # --- Ingestion & Parsing Settings ---
    # Supported document types for document_loader.py and their default parsers
    # Note: These are example mappings. Actual parsing logic might involve more dynamic selection.
    SUPPORTED_DOC_PARSERS = {
        "pdf": "src.ingestion.parsers.text_parser.TextParser",
        "md": "src.ingestion.parsers.text_parser.TextParser",
        "txt": "src.ingestion.parsers.text_parser.TextParser",
        "py": "src.ingestion.parsers.code_ast_parser.CodeASTParser",
        "java": "src.ingestion.parsers.code_ast_parser.CodeASTParser",
        "js": "src.ingestion.parsers.code_ast_parser.CodeASTParser",
        "html": "src.ingestion.parsers.tabular_data_parser.TabularDataParser", # HTML can contain tabular data
        "csv": "src.ingestion.parsers.tabular_data_parser.TabularDataParser",
        "xlsx": "src.ingestion.parsers.tabular_data_parser.TabularDataParser",
    }

    # Text Parser Settings
    TEXT_CHUNK_SIZE = int(os.getenv("TEXT_CHUNK_SIZE", "1024"))
    TEXT_CHUNK_OVERLAP = int(os.getenv("TEXT_CHUNK_OVERLAP", "128"))

    # Code AST Parser Settings
    CODE_LANGUAGES_SUPPORTED = ["python", "java", "javascript"]
    CODE_AST_GRANULARITY = os.getenv("CODE_AST_GRANULARITY", "function") # 'function', 'class', 'method', 'file'

    # Tabular Data Parser Settings
    TABULAR_MIN_ROWS = int(os.getenv("TABULAR_MIN_ROWS", "2"))
    TABULAR_MIN_COLS = int(os.getenv("TABULAR_MIN_COLS", "2"))
    # Max cells for an individual table to be processed (e.g., avoid parsing huge tables into LLM context)
    TABULAR_MAX_CELLS = int(os.getenv("TABULAR_MAX_CELLS", "1000"))

    # KG Extractor Settings
    KG_EXTRACTOR_MODEL = os.getenv("KG_EXTRACTOR_MODEL", "ollama:llama3") # Example: use a local LLM for extraction
    KG_RELATION_TYPES = ["causes", "prerequisites", "has_part", "uses", "defines", "calls", "implements", "returns", "is_a", "has_property"]
    KG_MAX_EXTRACTIONS_PER_CHUNK = int(os.getenv("KG_MAX_EXTRACTIONS_PER_CHUNK", "10"))


    # --- Multi-Modal Indexing & Storage Settings ---
    # Base directory for local storage, will be created if it doesn't exist
    STORAGE_DIR = os.getenv("STORAGE_DIR", "data/storage")

    # Vector Store Manager
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "CHROMA") # Options: CHROMA, FAISS, PINECONE, QDRANT
    CHROMA_PERSIST_DIRECTORY = os.path.join(STORAGE_DIR, "chroma_db")
    FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index.bin")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-hybrid-semantic-logical")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_chunks")

    # Graph Store Manager
    GRAPH_STORE_TYPE = os.getenv("GRAPH_STORE_TYPE", "NEO4J") # Options: NEO4J, ARANGODB, TINKERPOP (requires specific setup)
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    ARANGODB_HOST = os.getenv("ARANGODB_HOST", "http://localhost:8529")
    ARANGODB_USER = os.getenv("ARANGODB_USER", "root")
    ARANGODB_PASSWORD = os.getenv("ARANGODB_PASSWORD", "password")
    ARANGODB_DB_NAME = os.getenv("ARANGODB_DB_NAME", "rag_graph")

    # Structured Data Indexer (using SQLite for simplicity in prototype)
    STRUCTURED_DATA_DB_PATH = os.path.join(STORAGE_DIR, "structured_data.db")

    # Metadata Store (using SQLite for simplicity in prototype)
    METADATA_DB_PATH = os.path.join(STORAGE_DIR, "metadata.db")


    # --- Specialized Embedding Generation Settings ---
    # Default embedding dimension if not specified by model
    DEFAULT_EMBEDDING_DIMENSION = int(os.getenv("DEFAULT_EMBEDDING_DIMENSION", "768"))

    # Text Embedder
    TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # For API-based models like OpenAI, Cohere
    TEXT_EMBEDDING_API_KEY = os.getenv("TEXT_EMBEDDING_API_KEY")

    # Code Embedder (often specialized, e.g., CodeBERT, OpenAI's text-embedding-ada-002, or specific SBERT for code)
    CODE_EMBEDDING_MODEL = os.getenv("CODE_EMBEDDING_MODEL", "sentence-transformers/multi-qa-distilbert-cos-v1")
    CODE_EMBEDDING_API_KEY = os.getenv("CODE_EMBEDDING_API_KEY")

    # KG Embedder (e.g., PyKEEN models like TransE, ComplEx, RotatE)
    KG_EMBEDDING_MODEL = os.getenv("KG_EMBEDDING_MODEL", "pykeen/TransE_OpenBG75") # Example with OpenBG75 dataset
    # KG embedding typically happens internally; API key might be for graph storage if it provides embedding service
    KG_EMBEDDING_API_KEY = os.getenv("KG_EMBEDDING_API_KEY")


    # --- Hybrid Retrieval Orchestration Settings ---
    VECTOR_RETRIEVAL_TOP_K = int(os.getenv("VECTOR_RETRIEVAL_TOP_K", "5"))
    KG_RETRIEVAL_TOP_K = int(os.getenv("KG_RETRIEVAL_TOP_K", "3")) # Number of relevant graph paths/subgraphs/nodes
    STRUCTURED_DATA_RETRIEVAL_MAX_ROWS = int(os.getenv("STRUCTURED_DATA_RETRIEVAL_MAX_ROWS", "10"))

    # Re-ranking settings for fusing diverse results
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "10")) # Final number of top results after re-ranking

    # Weights for initial fusion if a simple weighted sum is used before re-ranking
    RETRIEVAL_WEIGHTS = {
        "vector": float(os.getenv("RETRIEVAL_WEIGHT_VECTOR", "0.5")),
        "kg": float(os.getenv("RETRIEVAL_WEIGHT_KG", "0.3")),
        "structured": float(os.getenv("RETRIEVAL_WEIGHT_STRUCTURED", "0.2")),
    }
    # Threshold for semantic similarity to be considered relevant
    SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.75"))
    # Max path length for KG queries
    KG_MAX_PATH_LENGTH = int(os.getenv("KG_MAX_PATH_LENGTH", "3"))


    # --- Dynamic Context Generation & LLM Integration Settings ---
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4096")) # Max tokens for the combined context fed to LLM
    MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "512")) # Max tokens for the LLM's generated response

    # LLM Adapter Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "OPENAI") # Options: OPENAI, ANTHROPIC, GEMINI, OLLAMA, HUGGINGFACE_LOCAL
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-opus-20240229")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3")
    # For local HuggingFace models, specify the path to the model directory
    HUGGINGFACE_LOCAL_MODEL_PATH = os.getenv("HUGGINGFACE_LOCAL_MODEL_PATH")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
    LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0"))
    LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.0"))

    @classmethod
    def _ensure_dir_exists(cls, path):
        """Ensures a directory exists, creating it if necessary, with basic error handling."""
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            # Log this as a warning or error in a real application
            print(f"WARNING: Could not create directory {path}. Error: {e}")

    @classmethod
    def validate_and_initialize(cls):
        """Performs basic validation and ensures necessary directories exist."""
        cls._ensure_dir_exists(cls.STORAGE_DIR)
        cls._ensure_dir_exists(cls.CHROMA_PERSIST_DIRECTORY) # For ChromaDB

        # Basic warnings for missing critical API keys/credentials for selected services
        if cls.VECTOR_STORE_TYPE.upper() == "PINECONE" and not (cls.PINECONE_API_KEY and cls.PINECONE_ENVIRONMENT):
            print("WARNING: Pinecone is selected as vector store, but PINECONE_API_KEY or PINECONE_ENVIRONMENT is not set.")
        if cls.GRAPH_STORE_TYPE.upper() == "NEO4J" and not (cls.NEO4J_URI and cls.NEO4J_USERNAME and cls.NEO4J_PASSWORD):
            print("WARNING: Neo4j is selected as graph store, but NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD is not set.")
        if cls.LLM_PROVIDER.upper() == "OPENAI" and not cls.OPENAI_API_KEY:
            print("WARNING: OpenAI is selected as LLM provider, but OPENAI_API_KEY is not set.")
        if cls.LLM_PROVIDER.upper() == "ANTHROPIC" and not cls.ANTHROPIC_API_KEY:
            print("WARNING: Anthropic is selected as LLM provider, but ANTHROPIC_API_KEY is not set.")
        if cls.LLM_PROVIDER.upper() == "GEMINI" and not cls.GEMINI_API_KEY:
            print("WARNING: Gemini is selected as LLM provider, but GEMINI_API_KEY is not set.")
        if cls.LLM_PROVIDER.upper() == "HUGGINGFACE_LOCAL" and not cls.HUGGINGFACE_LOCAL_MODEL_PATH:
            print("WARNING: HuggingFace local LLM is selected, but HUGGINGFACE_LOCAL_MODEL_PATH is not set.")

# Initialize configuration (e.g., create directories, print warnings) when this module is imported
Config.validate_and_initialize()