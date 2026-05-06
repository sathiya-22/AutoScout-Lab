import os
from pathlib import Path

class Config:
    """
    Centralized management for API keys, model names, vector store paths,
    and other environmental settings for the RAG prototype.
    """

    # --- API Keys ---
    # Retrieve API keys from environment variables for security.
    # Raise an error if critical API keys are not set.
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY") # Placeholder for future use

    # --- LLM Models ---
    # Define the names of the models to be used for various tasks.
    # Default to common models if not specified via environment variables.
    GENERATION_MODEL_NAME: str = os.getenv("GENERATION_MODEL_NAME", "gpt-4o-mini")
    RETRIEVAL_MODEL_NAME: str = os.getenv("RETRIEVAL_MODEL_NAME", "gpt-3.5-turbo")
    VERIFICATION_MODEL_NAME: str = os.getenv("VERIFICATION_MODEL_NAME", "gpt-4o-mini")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small") # OpenAI embedding model

    # --- Paths ---
    # Base directory for the project, allowing relative paths to be robust.
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # Data management paths
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DOCS_PATH: Path = DATA_DIR / "raw_docs"
    PROCESSED_CHUNKS_PATH: Path = DATA_DIR / "processed_chunks"
    VECTOR_STORE_PATH: Path = DATA_DIR / "vector_store"

    # Prompt engineering paths
    PROMPTS_DIR: Path = BASE_DIR / "prompts"
    RETRIEVAL_PROMPTS_PATH: Path = PROMPTS_DIR / "retrieval_prompts.py" # Assuming a Python module for prompts
    GENERATION_PROMPTS_PATH: Path = PROMPTS_DIR / "generation_prompts.py"
    VERIFICATION_PROMPTS_PATH: Path = PROMPTS_DIR / "verification_prompts.py"

    # Evaluation paths
    EVALUATION_DIR: Path = BASE_DIR / "evaluation"
    TEST_DATA_PATH: Path = EVALUATION_DIR / "test_data"
    EVALUATION_RESULTS_PATH: Path = EVALUATION_DIR / "results"

    # --- Data Processing Settings ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))
    # Supported file types for data loading
    SUPPORTED_FILE_TYPES: list = [".pdf", ".txt", ".md"]

    # --- Retrieval Settings ---
    TOP_K_SEMANTIC_RETRIEVAL: int = int(os.getenv("TOP_K_SEMANTIC_RETRIEVAL", 10))
    TOP_K_AFTER_RERANKING: int = int(os.getenv("TOP_K_AFTER_RERANKING", 5)) # Number of chunks to pass to generator
    # Threshold for correctness verification (e.g., a score from 0-1)
    CORRECTNESS_THRESHOLD: float = float(os.getenv("CORRECTNESS_THRESHOLD", 0.75))

    # --- Agentic Settings ---
    # Number of query rewrites/permutations to generate
    NUM_QUERY_PERMUTATIONS: int = int(os.getenv("NUM_QUERY_PERMUTATIONS", 3))

    # --- Environment/Logging ---
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development") # e.g., "development", "production", "testing"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO") # e.g., "DEBUG", "INFO", "WARNING", "ERROR"

    def __init__(self):
        self._validate_api_keys()
        self._ensure_directories_exist()

    def _validate_api_keys(self):
        """
        Validates that necessary API keys are present.
        """
        if not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it to your OpenAI API key."
            )
        # Add checks for other critical keys if they become mandatory

    def _ensure_directories_exist(self):
        """
        Ensures that all necessary directories for data storage and processing exist.
        """
        for path in [
            self.RAW_DOCS_PATH,
            self.PROCESSED_CHUNKS_PATH,
            self.VECTOR_STORE_PATH,
            self.PROMPTS_DIR, # Although prompts are files, the directory should exist
            self.EVALUATION_DIR,
            self.TEST_DATA_PATH,
            self.EVALUATION_RESULTS_PATH
        ]:
            path.mkdir(parents=True, exist_ok=True)

# Instantiate config for easy import
settings = Config()

# Example usage (for testing purposes, remove in final production code if not needed)
if __name__ == "__main__":
    print("--- Configuration Settings ---")
    print(f"OpenAI API Key Set: {'Yes' if settings.OPENAI_API_KEY else 'No'}")
    print(f"Generation Model: {settings.GENERATION_MODEL_NAME}")
    print(f"Vector Store Path: {settings.VECTOR_STORE_PATH}")
    print(f"Raw Docs Path: {settings.RAW_DOCS_PATH}")
    print(f"Chunk Size: {settings.CHUNK_SIZE}")
    print(f"Top K Semantic Retrieval: {settings.TOP_K_SEMANTIC_RETRIEVAL}")

    try:
        # Simulate missing API key for error handling test
        original_openai_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = ""
        Config()
    except ValueError as e:
        print(f"\nCaught expected error: {e}")
    finally:
        if original_openai_key:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        else:
            del os.environ["OPENAI_API_KEY"] # Clean up if it was not set originally
    print("\nConfiguration validation complete.")