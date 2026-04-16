from . import declarations
from . import compiler

# Re-export core DSL elements from declarations.py for easier import
from .declarations import (
    rule,
    condition,
    relationship,
    SpatialRule,
    SpatialCondition,
    SpatialRelationship,
    # Potentially other declarative elements like:
    # is_inside,
    # intersects,
    # distance_between,
    # ... these will be defined in declarations.py
)

# Re-export the compiler for easier access
from .compiler import compile_rules

__all__ = [
    "rule",
    "condition",
    "relationship",
    "SpatialRule",
    "SpatialCondition",
    "SpatialRelationship",
    "compile_rules",
    "declarations", # Expose the module itself too, if users want to access its internals
    "compiler",   # Expose the module itself too
]

# Basic initialization logic for the DSL package (if any)
# For now, simply indicating the package is ready.
# In a more complex scenario, this might involve registering default components
# or configuring logging for the DSL module.
# print("DSL package initialized.") # Removed for production code, useful for debugging during dev.