from .document_loader import DocumentLoader

# Define what symbols are exported when 'from ingestion import *' is used
__all__ = [
    "DocumentLoader",
]

# Basic error handling for core imports within the package
try:
    # Attempt to import key components to ensure they are available
    # and to catch any immediate import errors that might indicate a malformed package structure.
    # No additional logic for DocumentLoader needed here as it's directly exposed.
    pass
except ImportError as e:
    # Log or raise a more specific error if a critical component cannot be imported
    # This helps in debugging issues related to package structure or dependencies.
    # For __init__.py, directly raising might be appropriate as it indicates a fundamental problem.
    raise RuntimeError(f"Failed to initialize the ingestion package due to missing components: {e}") from e

# Further initialization logic could go here, e.g., setting up a logger,
# or configuring package-wide settings, but not specified in the current architecture.