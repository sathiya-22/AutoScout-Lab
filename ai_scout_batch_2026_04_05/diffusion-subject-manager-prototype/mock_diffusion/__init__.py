import logging

# Define the package version
__version__ = "0.1.0"

# Configure basic logging for the package, useful for debugging import issues.
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
_logger = logging.getLogger(__name__)

# Expose the core components of the Multi-Subject Latent Manager library.
# These imports assume the existence of the respective files in the 'src' directory
# relative to this __init__.py file.
try:
    from .src.subject_manager import MultiSubjectLatentManager
    from .src.spatial_biasing.adapter import SpatialBiasingAdapter
    _logger.info("Successfully imported MultiSubjectLatentManager and SpatialBiasingAdapter.")

except ImportError as e:
    # This block handles the edge case where core sub-modules are not found,
    # which might indicate an incomplete installation or a corrupted environment.
    _logger.critical(
        f"Failed to import core components for mock_diffusion: {e}. "
        "Please ensure 'src/subject_manager.py' and 'src/spatial_biasing/adapter.py' "
        "exist and are accessible in the package structure."
    )
    # Re-raise the exception as the package cannot function without its core components.
    raise

# Define what symbols are exposed when a user does `from mock_diffusion import *`.
__all__ = [
    "MultiSubjectLatentManager",
    "SpatialBiasingAdapter",
]

# Further package-level configurations or initializations can be added here.
# For instance, a global configuration object, default settings, or plugin registrations.
_logger.info("mock_diffusion package initialized.")