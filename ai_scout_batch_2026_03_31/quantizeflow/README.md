# QuantizeFlow: Adaptive Block-Quantization for Deep Learning

## Overview

QuantizeFlow is a Python library designed to bring efficient, dynamic, block-adaptive quantization schemes like IF4 (Int/Float 4) natively to deep learning frameworks. Existing frameworks such as PyTorch and TensorFlow currently lack high-level, performant support for these advanced ultra-low precision formats, requiring developers to implement complex, low-level kernels. QuantizeFlow abstracts this complexity, offering a user-friendly API for enhanced performance and reduced memory footprint.

## Problem Statement

The current deep learning ecosystem struggles with efficient implementation of dynamic, block-adaptive quantization. Schemes like IF4, which dynamically choose between FP4 and INT4 representation based on the statistical properties of data blocks, offer significant advantages in terms of memory and computational efficiency. However, integrating these techniques necessitates manual, low-level kernel development (e.g., CUDA, Triton), creating a high barrier to entry and slowing the adoption of superior quantization methods.

## Solution

QuantizeFlow addresses this by providing:

1.  **`AdaptiveQuantizedTensor`**: A custom tensor type that encapsulates packed 4-bit data, per-block scale factors, and critical block metadata (e.g., using the sign bit of the scale factor to indicate FP4/INT4).
2.  **High-Level Operations**: Functions like `adaptive_quant_matmul` and `adaptive_quant_linear` that seamlessly integrate block-adaptive quantization into your models.
3.  **Runtime Block Analysis**: A lightweight algorithm embedded within the kernels to dynamically determine the optimal FP4/INT4 representation for each 16-value block based on its value distribution.
4.  **Optimized Low-Level Kernels**: Custom CUDA/Triton kernels for GPUs (and optimized C++ intrinsics for CPUs) that dynamically interpret block metadata to perform mixed-mode (FP4 or INT4) Multiply-Accumulate operations within each block, accumulating into a higher-precision format.
5.  **Full Autograd Integration**: Supports quantized training with custom `torch.autograd.Function` implementations, allowing for end-to-end training of models using IF4.

## Features

*   **AdaptiveQuantizedTensor**: Custom tensor type for IF4 data, managing packed 4-bit values, per-block scales, and FP4/INT4 indicators.
*   **Seamless Integration**: High-level `adaptive_quant_matmul` and `adaptive_quant_linear` APIs.
*   **Dynamic Precision**: Runtime block analysis to select optimal FP4/INT4 for each data block.
*   **Performance**: Highly optimized C++/CUDA/Triton kernels for critical operations.
*   **Quantized Training**: Full autograd support for backpropagation through quantized operations.
*   **Cross-Platform**: GPU (CUDA/Triton) and CPU (C++ intrinsics) support.

## Installation

### Prerequisites

*   Python 3.8+
*   PyTorch 1.10+ (or JAX, depending on final backend choices)
*   For GPU support: NVIDIA CUDA Toolkit (compatible with your PyTorch installation)
*   For C++ compilation: `build-essential` (Linux) or equivalent developer tools (macOS/Windows).

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/quantizeflow.git
    cd quantizeflow
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Build and install the QuantizeFlow library:**
    This step will compile the C++/CUDA/Triton extensions.
    ```bash
    pip install .
    ```
    If you encounter compilation issues or wish to speed up the process on a multi-core machine, you can specify the number of parallel jobs:
    ```bash
    MAX_JOBS=4 pip install .
    ```

## Quick Start (Usage)

Here's a basic example demonstrating how to quantize a tensor and perform a quantized matrix multiplication:

```python
import torch
import quantizeflow.tensor as qf_tensor
import quantizeflow.ops as qf_ops

# Ensure CUDA is available if planning to use GPU
if not torch.cuda.is_available():
    print("CUDA not available, running on CPU.")
    device = 'cpu'
else:
    device = 'cuda'
    print(f"Running on {device}.")

# 1. Create a regular float tensor
input_tensor = torch.randn(64, 128, device=device, dtype=torch.float32)
weight_tensor = torch.randn(128, 256, device=device, dtype=torch.float32)

print(f"Input tensor shape: {input_tensor.shape}, device: {input_tensor.device}")
print(f"Weight tensor shape: {weight_tensor.shape}, device: {weight_tensor.device}")

# 2. Quantize the float weight tensor into an AdaptiveQuantizedTensor
# The `from_float` method performs the block analysis and packing.
print("\nQuantizing weight tensor...")
quantized_weight = qf_tensor.AdaptiveQuantizedTensor.from_float(weight_tensor, block_size=16)
print(f"Quantized weight created. Inner data shape: {quantized_weight._data.shape}") # _data is internal packed representation

# 3. Perform adaptive quantized matrix multiplication
print("Performing adaptive quantized matrix multiplication...")
output = qf_ops.adaptive_quant_matmul(input_tensor, quantized_weight)

print(f"\nQuantized MatMul Output:")
print(f"  Shape: {output.shape}")
print(f"  Device: {output.device}")
print(f"  Dtype: {output.dtype}")

# Example of using a quantized tensor in a linear layer (conceptually)
# linear_layer = qf_ops.adaptive_quant_linear(input_tensor, quantized_weight, bias=None)
# print(f"\nQuantized Linear Layer Output shape: {linear_layer.shape}")

```

For more detailed examples, including quantized training of a simple model, please refer to the `examples/` directory.

## Architecture

QuantizeFlow is structured as a Python library with a strong emphasis on performance-critical C++/CUDA/Triton extensions.

*   **Top-level (`quantizeflow/`)**: Project configuration, documentation, and build scripts (`setup.py`, `pyproject.toml`).
*   **Core Python Library (`quantizeflow/` package)**:
    *   `tensor.py`: Defines `AdaptiveQuantizedTensor`, encapsulating packed 4-bit data, scale factors, and block metadata.
    *   `ops.py`: Provides high-level APIs like `adaptive_quant_matmul` and `adaptive_quant_linear`, dispatching to low-level kernels.
    *   `autograd.py`: Implements custom `torch.autograd.Function` for quantized training.
    *   `utils.py`: Utility functions for data manipulation and kernel selection.
*   **Low-Level Extensions (`quantizeflow/_C/`)**: Highly optimized, device-specific implementations.
    *   `quantizeflow_ext.cpp`: Pybind11/TorchBind interface to expose C++/CUDA/CPU functions to Python.
    *   `common/if4_types.hpp`: Defines common IF4 data structures and metadata interpretation.
    *   **CUDA (`cuda/`)**: GPU-specific kernels.
        *   `block_analyzer.cu`: Runtime block analysis for optimal FP4/INT4 choice.
        *   `if4_quant.cuh/.cu`: IF4 quantization and dequantization kernels.
        *   `if4_gemm.cuh/.cu`: Core mixed-mode Multiply-Accumulate kernels.
    *   **Triton (`triton/`)**: High-performance Pythonic GPU kernels.
        *   `if4_matmul.py`, `if4_linear.py`: Triton implementations of quantized ops.
    *   **CPU (`cpu/`)**: Placeholder for optimized CPU implementations using intrinsics.
        *   `if4_quant_cpu.cpp`, `if4_gemm_cpu.cpp`: C++ implementations for CPU.

## Development and Testing

### Setting Up for Development

1.  Follow the installation steps.
2.  Install development-specific dependencies (if `requirements-dev.txt` exists):
    ```bash
    pip install -r requirements-dev.txt
    ```

### Running Tests

To ensure the correctness and stability of the library, run the comprehensive test suite:

```bash
pytest tests/
```
Specific test suites can be run by targeting their respective files, e.g., `pytest tests/test_cuda_kernels.py`.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.