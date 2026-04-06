from .base import RetrieverBase
from .vector_retriever import VectorRetriever
from .keyword_retriever import KeywordRetriever
from .graph_retriever import GraphRetriever
from .hybrid_retriever import HybridRetriever

# Optionally, define __all__ for explicit imports
__all__ = [
    "RetrieverBase",
    "VectorRetriever",
    "KeywordRetriever",
    "GraphRetriever",
    "HybridRetriever",
]

# Basic error handling for imports (uncommon for core internal modules unless optional)
try:
    from .vector_retriever import VectorRetriever
except ImportError as e:
    print(f"Warning: Could not import VectorRetriever. It might be due to missing dependencies. Error: {e}")
    VectorRetriever = None

try:
    from .keyword_retriever import KeywordRetriever
except ImportError as e:
    print(f"Warning: Could not import KeywordRetriever. It might be due to missing dependencies. Error: {e}")
    KeywordRetriever = None

try:
    from .graph_retriever import GraphRetriever
except ImportError as e:
    print(f"Warning: Could not import GraphRetriever. It might be due to missing dependencies. Error: {e}")
    GraphRetriever = None

try:
    from .hybrid_retriever import HybridRetriever
except ImportError as e:
    print(f"Warning: Could not import HybridRetriever. It might be due to missing dependencies. Error: {e}")
    HybridRetriever = None