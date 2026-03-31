"""
QuantizeFlow: A library for dynamic, block-adaptive quantization in deep learning.

Provides high-level API for AdaptiveQuantizedTensor types and operations
with optimized low-level kernels for IF4 (Int/Float 4) quantization.
"""

# Version of the QuantizeFlow library
__version__ = "0.0.1"

# Import core components to make them directly accessible under the quantizeflow namespace.
# This allows users to do:
# from quantizeflow import AdaptiveQuantizedTensor, adaptive_quant_matmul
from .tensor import AdaptiveQuantizedTensor
from .ops import adaptive_quant_matmul, adaptive_quant_linear

# Define __all__ to explicitly list the public API exposed when a user
# performs `from quantizeflow import *`.
__all__ = [
    "AdaptiveQuantizedTensor",
    "adaptive_quant_matmul",
    "adaptive_quant_linear",
]

# Note: Low-level C++/CUDA/Triton extensions are typically imported and used
# internally by the `ops.py` or `autograd.py` modules.
# This __init__.py file focuses on exposing the high-level Python API.
# Error handling for missing extensions will be managed by the modules
# that attempt to load and use them (e.g., ops.py) to provide more specific
# fallbacks or user messages.