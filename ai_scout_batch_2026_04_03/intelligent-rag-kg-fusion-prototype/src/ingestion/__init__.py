from .data_processor import DataProcessor
from .document_loader import DocumentLoader
from .knowledge_graph_builder import KnowledgeGraphBuilder
from .chunking_strategy import BaseChunkingStrategy, RecursiveCharacterTextChunker
from .metadata_extractor import MetadataExtractor

# Define package metadata
__version__ = "0.1.0"
__all__ = [
    "DataProcessor",
    "DocumentLoader",
    "KnowledgeGraphBuilder",
    "BaseChunkingStrategy",
    "RecursiveCharacterTextChunker",
    "MetadataExtractor"
]

# You can add a simple initialization logic here if needed,
# for example, to set up logging or load configuration.
# For this prototype, an empty init is sufficient, with key components imported.

# Example of a basic setup (optional, for demonstration)
try:
    # This might be used if there's a global ingestion configuration or setup
    # that needs to happen when the package is imported.
    # For now, we keep it minimal.
    pass
except Exception as e:
    # Basic error handling for package initialization
    print(f"Error initializing ingestion package: {e}")