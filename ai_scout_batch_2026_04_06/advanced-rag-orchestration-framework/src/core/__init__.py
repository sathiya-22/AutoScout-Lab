```python
"""
The `core` package provides the fundamental building blocks for the Advanced RAG Orchestration Framework.

It defines:
-   `orchestrator`: The central control plane for defining and executing RAG pipelines.
-   `interfaces`: Abstract Base Classes (ABCs) for all pluggable modules, ensuring standardized inputs/outputs.
-   `models`: Pydantic-based data models for type-safe data flow through the pipeline.

This package serves as the backbone, enabling modularity, extensibility, and clarity across the framework.
"""

# Define the package version
__version__ = "0.1.0"

# Expose key components from submodules for easier access
# For example, you can import RAGOrchestrator directly from src.core
try:
    from .orchestrator import RAGOrchestrator
except ImportError as e:
    raise ImportError(f"Failed to import RAGOrchestrator from src.core.orchestrator: {e}")

# Expose all defined interfaces
try:
    from .interfaces import (
        QueryProcessor,
        Retriever,
        Reranker,
        EvidenceCompressor,
        Generator,
        AgenticModule,
    )
except ImportError as e:
    raise ImportError(f"Failed to import interfaces from src.core.interfaces: {e}")

# Expose all data models
try:
    from .models import (
        Query,
        Document,
        RetrievalResult,
        Context,
        PipelineResult,
    )
except ImportError as e:
    raise ImportError(f"Failed to import models from src.core.models: {e}")


# Define what is exposed when a user does `from src.core import *`
__all__ = [
    "RAGOrchestrator",
    # Interfaces
    "QueryProcessor",
    "Retriever",
    "Reranker",
    "EvidenceCompressor",
    "Generator",
    "AgenticModule",
    # Models
    "Query",
    "Document",
    "RetrievalResult",
    "Context",
    "PipelineResult",
]
```