__version__ = "0.1.0"

# This file marks the 'src' directory as a Python package.
# It can be used for package-wide initialization, exposing
# key components for easier access, or defining package metadata.

# For this advanced RAG orchestration framework, we prioritize
# modularity and clear separation of concerns. Therefore,
# specific components (e.g., RAGOrchestrator, specific retrievers)
# are intended to be imported directly from their respective
# sub-modules (e.g., `from src.core.orchestrator import RAGOrchestrator`).

# This __init__.py primarily serves to define the package
# structure and its version. No complex logic or extensive
# imports are placed here to maintain a clean and explicit
# import experience for developers using the framework.

# You might consider importing a global logging configuration
# or essential utility functions if they are truly package-wide,
# but for now, we assume `src.utils` handles such setup explicitly.

# Example of a potential global configuration/utility import (optional):
# from .utils import setup_logging
# setup_logging() # If logging needs to be configured immediately on package import