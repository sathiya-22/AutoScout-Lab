import os
from pathlib import Path

# --- Base Project Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for _dir in [DATA_DIR, DOCUMENTS_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# --- API Keys (loaded from environment variables for security) ---
# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Hugging Face
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Model Configuration ---
# Embedding Model
EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "OPENAI") # Options: "OPENAI", "HUGGINGFACE"
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
HUGGINGFACE_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 1536 if EMBEDDING_MODEL_TYPE == "OPENAI" else 384 # Ada-002: 1536, MiniLM: 384

# Large Language Model (LLM)
LLM_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE", "OPENAI") # Options: "OPENAI", "ANTHROPIC", "HUGGINGFACE"
OPENAI_LLM_MODEL_NAME = "gpt-4-turbo-preview"
ANTHROPIC_LLM_MODEL_NAME = "claude-3-opus-20240229"
HUGGINGFACE_LLM_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf" # Requires HUGGINGFACE_API_KEY or local setup

LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096

# --- Database Configuration ---
# Vector Store
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "CHROMA") # Options: "CHROMA", "FAISS", "WEAVIATE"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"

# Weaviate specific configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
WEAVIATE_CLASS_NAME = "RAGDocumentChunk"

# Graph Database
GRAPH_DB_TYPE = os.getenv("GRAPH_DB_TYPE", "NETWORKX_SQLITE") # Options: "NETWORKX_SQLITE", "NEO4J"
NETWORKX_SQLITE_PATH = DATA_DIR / "graph.db" # For NetworkX persistence using SQLite

# Neo4j specific configuration (for more robust graph storage)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# --- Ingestion & Preprocessing ---
# Chunking Strategy
DEFAULT_CHUNK_SIZE = 1000 # Default chunk size in characters
DEFAULT_CHUNK_OVERLAP = 100 # Overlap for consecutive chunks
SENTENCE_WINDOW_SIZE = 3 # For advanced context-aware chunking (e.g., surrounding sentences)

# Graph Extraction
GRAPH_EXTRACTION_MODEL_NAME = OPENAI_LLM_MODEL_NAME # Can be a specific LLM or rule-based

# --- Retrieval Parameters ---
DEFAULT_VECTOR_TOP_K = 5
DEFAULT_KEYWORD_TOP_K = 3
DEFAULT_GRAPH_HOPS = 2 # Max hops for graph traversal
DEFAULT_LOGICAL_CONTEXT_WINDOW = 3 # Number of surrounding chunks to consider for logical context

# Hybrid Retrieval Weights (sum should ideally be 1, but can be adjusted for ranking)
HYBRID_RETRIEVAL_WEIGHTS = {
    "vector": 0.4,
    "keyword": 0.2,
    "graph": 0.3,
    "logical": 0.1,
}

# --- Agent-Driven Query Processing ---
MAX_AGENT_ITERATIONS = 5 # Max number of steps the agent can take
AGENT_REASONING_MODEL = OPENAI_LLM_MODEL_NAME # LLM used by the agent for decision making

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = LOGS_DIR / "rag_prototype.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --- Default Document Loader Settings ---
MAX_FILE_SIZE_MB = 10 # Maximum file size for documents to be loaded

# --- Fallback/Default Settings ---
DEFAULT_LLM_FALLBACK = "gpt-3.5-turbo" # If primary LLM fails or is unavailable
DEFAULT_EMBEDDING_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2" # If primary embedding fails

# --- Environment Check ---
def check_environment_variables():
    """Checks for critical environment variables and logs warnings if not set."""
    if LLM_MODEL_TYPE == "OPENAI" and not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY is not set. OpenAI LLM and embedding models may not function.")
    if LLM_MODEL_TYPE == "ANTHROPIC" and not ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY is not set. Anthropic LLMs may not function.")
    if (LLM_MODEL_TYPE == "HUGGINGFACE" or EMBEDDING_MODEL_TYPE == "HUGGINGFACE") and not HUGGINGFACE_API_KEY:
        print("WARNING: HUGGINGFACE_API_KEY is not set. Hugging Face models requiring authentication may not function.")
    if GRAPH_DB_TYPE == "NEO4J" and (not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD):
        print("WARNING: Neo4j connection details (NEO4J_URI, USER, PWD) are not fully set. Neo4j graph DB may not function.")
    if VECTOR_DB_TYPE == "WEAVIATE" and not WEAVIATE_URL:
        print("WARNING: WEAVIATE_URL is not set. Weaviate vector DB may not function.")

# Optionally call this during application startup
check_environment_variables()