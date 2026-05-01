from .core import RLGuard
from .config import RLGuardConfig

__all__ = [
    "RLGuard",
    "RLGuardConfig",
]

# Basic package versioning, could be managed by a separate tool later
__version__ = "0.1.0"

# Optional: Initialize logging for the package
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())