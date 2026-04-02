import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Literal, Optional

# Define the root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    """
    Centralized configuration for the Personalized Multimodal Evidence Grounding (PMEG) Framework.
    Settings are loaded from environment variables, and optionally from a .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- General Application Settings ---
    APP_NAME: str = "PMEG Framework"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # --- Database Configurations ---
    # Relational Database (for metadata, file status, user profiles)
    SQLALCHEMY_DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/pmeg_metadata.db"

    # Vector Database (for cross-modal embeddings)
    VECTOR_DB_TYPE: Literal["PINECONE", "WEAVIATE", "FAISS_LOCAL", "QDRANT"] = "FAISS_LOCAL"
    # Pinecone
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "pmeg-embeddings"
    # Weaviate
    WEAVIATE_URL: Optional[str] = None
    WEAVIATE_API_KEY: Optional[str] = None
    # Qdrant
    QDRANT_HOST: Optional[str] = None
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "pmeg_embeddings"
    # FAISS (Local)
    FAISS_INDEX_PATH: str = f"{BASE_DIR}/data/faiss_index.bin"

    # Graph Database (for Semantic Entity Graph)
    GRAPH_DB_TYPE: Literal["NEO4J", "ARANGODB", "TINKERPOP"] = "NEO4J"
    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: Optional[str] = None
    NEO4J_DATABASE: str = "neo4j"
    # ArangoDB (example, not fully implemented for prototype)
    ARANGODB_HOST: str = "localhost"
    ARANGODB_PORT: int = 8529
    ARANGODB_USER: str = "root"
    ARANGODB_PASSWORD: Optional[str] = None
    ARANGODB_DB_NAME: str = "pmeg_graph"

    # --- Perception Agent Configurations ---
    # OCR Agent
    OCR_ENGINE: Literal["TESSERACT", "GOOGLE_VISION_API", "AZURE_COGNITIVE_SERVICE"] = "TESSERACT"
    TESSERACT_PATH: str = "/usr/bin/tesseract" # Adjust for your Tesseract installation
    GOOGLE_VISION_API_KEY_PATH: Optional[str] = None # Path to service account JSON file
    AZURE_OCR_ENDPOINT: Optional[str] = None
    AZURE_OCR_KEY: Optional[str] = None

    # Image Captioning Agent
    IMAGE_CAPTIONING_MODEL: Literal["BLIP-LARGE", "LLAVA", "OPENAI_CLIP"] = "BLIP-LARGE"
    IMAGE_CAPTIONING_DEVICE: Literal["cpu", "cuda"] = "cpu" if not os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cuda"
    IMAGE_CAPTIONING_MODEL_PATH: Optional[str] = None # For local models, e.g., Hugging Face model ID or local path
    OPENAI_API_KEY: Optional[str] = None # For OpenAI models like CLIP, DALL-E (not used for captioning directly, but if general OpenAI access is needed)

    # Audio Transcription Agent
    AUDIO_TRANSCRIPTION_MODEL: Literal["WHISPER_TINY", "WHISPER_BASE", "GOOGLE_SPEECH_TO_TEXT_API"] = "WHISPER_TINY"
    WHISPER_MODEL_PATH: Optional[str] = None # For local Whisper models, e.g., Hugging Face model ID or local path
    GOOGLE_SPEECH_TO_TEXT_API_KEY_PATH: Optional[str] = None # Path to service account JSON file

    # Video Analysis Agent
    VIDEO_ANALYSIS_MODEL: Literal["YOLOv8", "MEDIAPIPE_POSE", "MMDETECTION"] = "YOLOv8"
    VIDEO_ANALYSIS_DEVICE: Literal["cpu", "cuda"] = "cpu" if not os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cuda"
    VIDEO_ANALYSIS_MODEL_PATH: Optional[str] = None # For local models, e.g., pre-trained YOLO weights

    # Document Parsing Agent
    DOCUMENT_PARSER_LIB: Literal["UNSTRUCTURED", "LANGCHAIN_PARSERS"] = "UNSTRUCTURED"
    # Add any specific config for these libs if needed

    # --- Cross-Modal Embedding & Alignment Configurations ---
    MULTIMODAL_EMBEDDING_MODEL: Literal["CLIP-ViT-B-32", "MPNET_BASE_V2", "OPENAI_TEXT_EMBEDDING_ADA_002", "MISTRAL_INSTRUCT_EMBEDDING"] = "CLIP-ViT-B-32"
    EMBEDDING_MODEL_PATH: Optional[str] = None # Path to local model or Hugging Face ID
    EMBEDDING_MODEL_DIMENSION: int = 512 # Default for CLIP-ViT-B-32, adjust based on model
    ALIGNMENT_TRAINER_CONFIG_PATH: str = f"{BASE_DIR}/config/alignment_hyperparams.json" # Path to alignment training config
    # For models requiring specific APIs (e.g., OpenAI)
    OPENAI_API_KEY: Optional[str] = None
    OLLAMA_API_BASE_URL: Optional[str] = "http://localhost:11434" # For local LLMs like Mistral embedding

    # --- File System Monitoring (Ingestion) ---
    MONITORED_DIRECTORIES: List[str] = [f"{BASE_DIR}/data/personal_files"] # Directories to watch
    SCAN_INTERVAL_SECONDS: int = 300 # How often to scan for new/modified files
    MAX_QUEUE_SIZE: int = 1000 # Max files in the processing queue
    SUPPORTED_FILE_TYPES: Dict[str, List[str]] = {
        "text": ["txt", "md", "pdf", "docx", "pptx", "xlsx", "csv"],
        "image": ["jpg", "jpeg", "png", "gif", "webp"],
        "audio": ["mp3", "wav", "flac"],
        "video": ["mp4", "mov", "avi", "mkv"],
    }
    PDF_OCR_ENABLED: bool = True # Whether to OCR PDFs if text extraction fails or is incomplete

    # --- Contextual Grounding & Iterative Synthesis ---
    # LLM for reasoning and synthesis
    LLM_MODEL: Literal["GPT-4o", "CLAUDE_3_OPUS", "MISTRAL_LARGE", "OLLAMA_MIXTRAL"] = "GPT-4o"
    ANTHROPIC_API_KEY: Optional[str] = None
    # Use OPENAI_API_KEY for GPT models
    # OLLAMA_API_BASE_URL for Ollama models

    # --- Data Paths ---
    DATA_DIR: str = f"{BASE_DIR}/data"
    PROCESSED_DATA_DIR: str = f"{DATA_DIR}/processed"
    MODELS_DIR: str = f"{DATA_DIR}/models"
    LOGS_DIR: str = f"{BASE_DIR}/logs"

    # Ensure data directories exist
    def _ensure_dirs(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        for path in self.MONITORED_DIRECTORIES:
            os.makedirs(path, exist_ok=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ensure_dirs()

# Initialize settings
settings = Settings()

# Example usage (for local testing, can be removed in production config)
if __name__ == "__main__":
    print(f"App Name: {settings.APP_NAME}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"SQL Alchemy DB: {settings.SQLALCHEMY_DATABASE_URL}")
    print(f"Vector DB Type: {settings.VECTOR_DB_TYPE}")
    if settings.VECTOR_DB_TYPE == "FAISS_LOCAL":
        print(f"FAISS Index Path: {settings.FAISS_INDEX_PATH}")
    print(f"Graph DB Type: {settings.GRAPH_DB_TYPE}")
    print(f"Neo4j URI: {settings.NEO4J_URI}")
    print(f"OCR Engine: {settings.OCR_ENGINE}")
    print(f"Image Captioning Device: {settings.IMAGE_CAPTIONING_DEVICE}")
    print(f"Monitored Directories: {settings.MONITORED_DIRECTORIES}")
    print(f"Multimodal Embedding Model: {settings.MULTIMODAL_EMBEDDING_MODEL}")
    print(f"LLM Model for Synthesis: {settings.LLM_MODEL}")
    print(f"Data Directory: {settings.DATA_DIR}")

    # You can override settings for testing:
    # os.environ["PMEG_DEBUG"] = "true"
    # new_settings = Settings()
    # print(f"New Debug Mode (after env override): {new_settings.DEBUG}")

    # Accessing sensitive info from .env:
    # Ensure you have OPENAI_API_KEY="sk-..." in your .env file
    # print(f"OpenAI API Key (first 5 chars): {settings.OPENAI_API_KEY[:5] if settings.OPENAI_API_KEY else 'Not Set'}")