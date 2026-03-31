import sys
import os

# This `__init__.py` file is responsible for loading the compiled C++/CUDA/CPU
# extension module for QuantizeFlow.

# Placeholder for the compiled C++ / CUDA extension module.
# This variable will hold the loaded extension if successful, or None otherwise.
_quantizeflow_cpp_module = None

# Attempt to load the compiled C++/CUDA/CPU extension.
# It is expected that the `setup.py` (or equivalent build system, potentially
# leveraging `torch.utils.cpp_extension`) will build a shared library
# (e.g., `_quantizeflow_ext.so` on Linux/macOS, `_quantizeflow_ext.pyd` on Windows)
# and place it within this `quantizeflow/_C/` directory.
# We then import it as a standard Python module relative to the current package.
try:
    from . import _quantizeflow_ext
    _quantizeflow_cpp_module = _quantizeflow_ext
except ImportError as e:
    # If the extension is not found, it means it hasn't been built,
    # is not in the expected path, or the build process failed.
    # This is a common scenario during development before a full installation
    # with compilation. We print a warning to inform the user but allow the
    # rest of the Python package to load, though low-level C++ functionalities
    # will be unavailable.
    sys.stderr.write(
        f"QuantizeFlow Warning: Failed to load low-level C++ extension "
        f"(_quantizeflow_ext). This is expected if QuantizeFlow has not been "
        f"built with C++ extensions (e.g., during development or if `setup.py` "
        f"was run without compilation flags). Please compile it for full "
        f"functionality. Error details: {e}\n"
    )
    sys.stderr.flush()
except Exception as e:
    # Catch any other unexpected errors that might occur during the import
    # process (e.g., issues within the C++ module itself upon loading due
    # to corrupted binaries or runtime environment problems).
    sys.stderr.write(
        f"QuantizeFlow Error: An unexpected error occurred while loading "
        f"the QuantizeFlow C++ extension: {e}\n"
    )
    sys.stderr.flush()

# Define __all__ to explicitly state what symbols are intended to be exported
# when `from quantizeflow._C import *` is used.
# In this case, we primarily want to expose the loaded C++ module (or its absence).
__all__ = ["_quantizeflow_cpp_module"]

# If there were any pure Python modules residing within the `_C` subdirectory
# (e.g., common utility Python files that are part of the low-level interface
# but not compiled), they would be imported here as well.
# Based on the architecture notes, `_C` is predominantly for native extensions,
# so direct Python modules here are less likely for this specific prototype.