import logging

# Configure logging for the core module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Avoid adding handlers here to let the root logger or application config handle it,
# but provide a NullHandler to prevent "No handler could be found for logger..." messages
logger.addHandler(logging.NullHandler())

# Import submodules to make them part of the core package
# This allows users to do `import src.core.primitives` or `from src.core import primitives`
from . import primitives
from . import operations

# Optionally, expose common primitives and operations directly under the `src.core` namespace
# for easier access without having to drill down into `primitives.` or `operations.`
try:
    from .primitives import (
        Point, Vector, Ray, Line, Plane,
        BoundingBox, Sphere, Triangle,
        MeshPrimitive, SceneObject,
        EPSILON # A common geometric tolerance
    )
    from .operations import (
        intersects, distance, contains,
        transform, project,
        # Add more core operations as they are defined
    )
except ImportError as e:
    logger.error(f"Failed to import core geometric components: {e}")
    # Re-raise to indicate a critical setup failure
    raise

logger.info("src.core package initialized successfully.")

# Define __all__ for explicit control over what `from src.core import *` imports
__all__ = [
    "primitives",
    "operations",
    "Point",
    "Vector",
    "Ray",
    "Line",
    "Plane",
    "BoundingBox",
    "Sphere",
    "Triangle",
    "MeshPrimitive",
    "SceneObject",
    "EPSILON",
    "intersects",
    "distance",
    "contains",
    "transform",
    "project",
]