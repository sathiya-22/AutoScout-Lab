import logging
import os
import uuid
import yaml
from enum import Enum

class DocumentType(Enum):
    """Enumerates the types of documents supported by the system."""
    TEXT = "text"
    CODE = "code"
    TABULAR = "tabular"
    PDF = "pdf"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"

class RetrievalStrategy(Enum):
    """Enumerates the different retrieval strategies available."""
    VECTOR_SEARCH = "vector_search"
    GRAPH_QUERY = "graph_query"
    STRUCTURED_LOOKUP = "structured_lookup"
    HYBRID = "hybrid" # Orchestrator decides best combination
    KEYWORD_MATCH = "keyword_match" # Potentially useful as a fallback or pre-filter

def setup_logging(name: str = "rag_system", level=logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance for the application.

    Args:
        name (str): The name of the logger.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if setup_logging is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Optional: File handler
        # log_dir = "logs"
        # os.makedirs(log_dir, exist_ok=True)
        # fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        # fh.setLevel(level)
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)

    return logger

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads configuration settings from a YAML file.

    Args:
        config_path (str): The path to the configuration YAML file.

    Returns:
        dict: A dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration file: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred loading configuration: {e}")

def generate_unique_id() -> str:
    """
    Generates a universally unique identifier (UUID).

    Returns:
        str: A UUID string.
    """
    return str(uuid.uuid4())

def sanitize_text(text: str) -> str:
    """
    Performs basic text sanitization (e.g., stripping leading/trailing whitespace).

    Args:
        text (str): The input text.

    Returns:
        str: The sanitized text.
    """
    if not isinstance(text, str):
        return str(text) # Attempt to convert non-string types to string
    return text.strip()

def chunk_text_by_length(text: str, max_chunk_length: int, overlap: int = 0) -> list[str]:
    """
    Simple utility to chunk text based on a maximum character length.
    This is a basic chunking and might be superseded by more advanced parsers.

    Args:
        text (str): The input text to chunk.
        max_chunk_length (int): The maximum character length of each chunk.
        overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        return []
    if max_chunk_length <= 0:
        raise ValueError("max_chunk_length must be positive.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= max_chunk_length:
        logger = setup_logging(__name__)
        logger.warning(
            f"Overlap ({overlap}) is greater than or equal to max_chunk_length "
            f"({max_chunk_length}). This may lead to redundant chunks."
        )

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_length, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start += max_chunk_length - overlap
        # Ensure start doesn't go backwards if overlap is too large relative to remaining text
        if start < 0:
            start = 0
    return chunks

# Initialize a default logger for utils module
logger = setup_logging(__name__)

if __name__ == '__main__':
    # Example usage of utilities
    logger.info("--- Utils Module Demo ---")

    # 1. Configuration loading
    # Create a dummy config.yaml for demonstration
    dummy_config_content = """
    app:
      name: Hybrid RAG System
      version: 0.1.0
    database:
      vector_store_url: "http://localhost:8108"
      graph_db_url: "bolt://localhost:7687"
    llm:
      api_key: "sk-xxxxxx"
      model_name: "gpt-4"
    """
    with open("config.yaml", "w") as f:
        f.write(dummy_config_content)

    try:
        app_config = load_config("config.yaml")
        logger.info(f"Loaded config: {app_config}")
        logger.info(f"LLM model name: {app_config.get('llm', {}).get('model_name')}")
    except (FileNotFoundError, yaml.YAMLError, Exception) as e:
        logger.error(f"Failed to load config: {e}")

    # 2. Unique ID generation
    id1 = generate_unique_id()
    id2 = generate_unique_id()
    logger.info(f"Generated unique ID 1: {id1}")
    logger.info(f"Generated unique ID 2: {id2}")

    # 3. DocumentType Enum
    doc_type = DocumentType.CODE
    logger.info(f"Document Type: {doc_type.name} (value: {doc_type.value})")

    # 4. RetrievalStrategy Enum
    ret_strat = RetrievalStrategy.STRUCTURED_LOOKUP
    logger.info(f"Retrieval Strategy: {ret_strat.name} (value: {ret_strat.value})")

    # 5. Text sanitization
    dirty_text = "  Hello, World! \n"
    clean_text = sanitize_text(dirty_text)
    logger.info(f"Dirty text: '{dirty_text}'")
    logger.info(f"Clean text: '{clean_text}'")

    # 6. Basic chunking
    long_text = "This is a very long sentence that needs to be broken into smaller pieces. We will demonstrate how our utility function can split text based on a specified maximum length and an optional overlap. This helps in processing large documents."
    chunks = chunk_text_by_length(long_text, max_chunk_length=30, overlap=10)
    logger.info(f"Original text length: {len(long_text)}")
    logger.info(f"Chunked into {len(chunks)} pieces:")
    for i, chunk in enumerate(chunks):
        logger.info(f"  Chunk {i+1} (len {len(chunk)}): '{chunk}'")

    # Clean up dummy config
    if os.path.exists("config.yaml"):
        os.remove("config.yaml")