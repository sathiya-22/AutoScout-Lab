```python
# src/__init__.py

__version__ = "0.1.0-alpha"

# This __init__.py file marks the 'src' directory as a Python package.
# It serves as a central entry point for the 'geometric-reasoning-dsl-prototype' library.
#
# Following the principles of Modularity & Decoupling, this top-level __init__.py
# remains minimal. Components are generally expected to be imported directly
# from their specific sub-packages (e.g., `from src.dsl import declarations`).
# This design choice helps maintain clear separation of concerns and avoids
# polluting the top-level namespace, aligning with the goal of reducing
# architectural coupling.

# Future versions might expose highly stable, core interfaces here for convenience
# if they become ubiquitous entry points for end-users.
# For example:
# from .dsl.declarations import rule, is_inside, intersects, contains
# from .oracle.validation_engine import ValidationEngine

# For the initial prototype, explicit imports from sub-packages are encouraged
# to highlight the modular structure.
```