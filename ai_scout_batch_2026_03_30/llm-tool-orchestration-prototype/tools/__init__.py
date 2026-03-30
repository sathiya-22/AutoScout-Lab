```python
# tools/__init__.py

# Import key components to make them easily accessible when importing the 'tools' package.

try:
    from .tool_registry import ToolRegistry
except ImportError as e:
    # Basic error handling: if ToolRegistry cannot be imported, it's a critical issue
    # for the system's architecture. Print an informative message and re-raise.
    print(f"ERROR: Could not import ToolRegistry. Please ensure 'tools/tool_registry.py' exists "
          f"and defines a 'ToolRegistry' class. Details: {e}")
    raise

try:
    from .sandbox_executor import SandboxExecutor
except ImportError as e:
    # Similar critical error handling for SandboxExecutor.
    print(f"ERROR: Could not import SandboxExecutor. Please ensure 'tools/sandbox_executor.py' exists "
          f"and defines a 'SandboxExecutor' class. Details: {e}")
    raise

# The 'tool_definitions' module is expected to contain Pydantic schemas for various tools.
# It's typically imported directly by other modules when they need to define, register,
# or validate specific tool schemas (e.g., 'from tools.tool_definitions import SearchToolSchema').
# We generally don't expose every specific tool schema at the top-level of the 'tools' package,
# unless there's a common base class or a factory function that should be universally accessible.

# Define __all__ to explicitly list the public symbols exposed by the 'tools' package
# when 'from tools import *' is used. This also serves as clear documentation of the package's public API.
__all__ = [
    "ToolRegistry",
    "SandboxExecutor",
]
```