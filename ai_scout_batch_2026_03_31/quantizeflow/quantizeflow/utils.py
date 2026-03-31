```python
import torch
import warnings
from typing import Tuple, Optional, Callable

# Constants for IF4 blocks
IF4_BLOCK_SIZE: int = 16

# --- Internal state for C++/CUDA/Triton extensions ---
_quantizeflow_ext = None
_use_triton_kernels: Optional[bool] = None # Will be set to True/False/None (unset)


def _get_ext_module():
    """Lazily loads and returns the C++/CUDA/CPU extension module."""
    global _quantizeflow_ext
    if _quantizeflow_ext is None:
        try:
            # Attempt to import the compiled extension
            import quantizeflow._C as _quantizeflow_ext_mod
            _quantizeflow_ext = _quantizeflow_ext_mod
        except ImportError:
            warnings.warn(
                "QuantizeFlow C++/CUDA/CPU extensions not found. "
                "Falling back to basic Python logic if available, "
                "but performance will be severely degraded. "
                "Please ensure the library is properly installed and compiled."
            )
            _quantizeflow_ext = None # Ensure it's explicitly None on failure
    return _quantizeflow_ext

def _has_cuda() -> bool:
    """Checks if CUDA is available and the extension module supports it."""
    ext = _get_ext_module()
    return torch.cuda.is_available() and ext is not None and hasattr(ext, 'quantize_if4_cuda')

def _has_triton() -> bool:
    """Checks if Triton kernels are enabled and available."""
    global _use_triton_kernels
    if _use_triton_kernels is None:
        try:
            import triton
            import triton.language as tl
            # Check if Triton GPU support is available
            _use_triton_kernels = torch.cuda.is_available()
            if not _use_triton_kernels:
                warnings.warn("Triton requires CUDA, but no CUDA devices found. Triton kernels disabled.")
        except ImportError:
            warnings.warn("Triton not found. Triton kernels disabled.")
            _use_triton_kernels = False
        except Exception as e:
            warnings.warn(f"Error checking Triton availability: {e}. Triton kernels disabled.")
            _use_triton_kernels = False
    return _use_triton_kernels

def set_use_triton_kernels(enable: bool):
    """Globally enables or disables the use of Triton kernels."""
    global _use_triton_kernels
    _use_triton_kernels = enable
    if enable and not _has_triton(): # _has_triton will re-evaluate and warn if needed
        warnings.warn("Attempted to enable Triton kernels, but Triton is not available or supported.")
        _use_triton_kernels = False

# --- Block type encoding/decoding using the scale factor's sign bit ---
# Convention: Positive scale -> INT4, Negative scale -> FP4
# The actual scale value is always the absolute value.

def encode_block_type(scales: torch.Tensor, is_fp4_block: torch.BoolTensor) -> torch.Tensor:
    """
    Encodes the block type (FP4 or INT4) into the sign bit of the scale factor.
    A negative scale indicates an FP4 block.
    """
    if scales.shape != is_fp4_block.shape:
        raise ValueError("Scales and is_fp4_block must have the same shape.")
    
    # Ensure scales are positive before applying sign for encoding
    abs_scales = torch.abs(scales)
    
    # Where is_fp4_block is True, make the scale negative
    encoded_scales = torch.where(is_fp4_block, -abs_scales, abs_scales)
    return encoded_scales

def decode_block_type(encoded_scales: torch.Tensor) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Decodes the block type (FP4 or INT4) from the sign bit of the scale factor.
    Returns the actual (positive) scale factor and a boolean tensor indicating if it's an FP4 block.
    """
    is_fp4_block = encoded_scales < 0
    actual_scales = torch.abs(encoded_scales)
    return actual_scales, is_fp4_block

def is_fp4_block_from_encoded_scale(encoded_scales: torch.Tensor) -> torch.BoolTensor:
    """
    Returns a boolean tensor indicating if each block is FP4 based on its encoded scale.
    """
    return encoded_scales < 0

# --- Core Quantization/Dequantization kernel dispatchers ---

def quantize_to_if4_kernel(
    input_tensor: torch.Tensor,
    block_size: int = IF4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    High-level dispatcher for quantizing a float32 tensor to the IF4 format.
    Returns packed_data (uint8), scales (float), and block_metadata (float, implicitly encoded).
    """
    if not input_tensor.is_floating_point():
        raise TypeError(f"Input tensor must be floating point, got {input_tensor.dtype}")
    if input_tensor.dim() not in [1, 2, 3, 4]: # Common tensor dims for quantization
        warnings.warn(f"Quantizing a {input_tensor.dim()}-D tensor. Block analysis assumes flat blocks.")

    ext = _get_ext_module()
    device = input_tensor.device

    if device.type == 'cuda' and _has_triton():
        try:
            # Import Triton kernels dynamically to avoid circular dependencies and only if enabled
            from quantizeflow._C.triton import if4_quant as triton_if4_quant
            packed_data, scales, block_metadata = triton_if4_quant.quantize_if4_triton(
                input_tensor, block_size
            )
            return packed_data, scales, block_metadata
        except Exception as e:
            warnings.warn(f"Triton IF4 quantization failed: {e}. Falling back to CUDA/CPU.")
            # If Triton fails, try CUDA or CPU
            pass

    if device.type == 'cuda' and _has_cuda():
        if ext is None or not hasattr(ext, 'quantize_if4_cuda'):
            raise RuntimeError("CUDA extensions are not loaded or 'quantize_if4_cuda' is not available.")
        return ext.quantize_if4_cuda(input_tensor, block_size)
    elif device.type == 'cpu':
        if ext is None or not hasattr(ext, 'quantize_if4_cpu'):
            raise RuntimeError("CPU extensions are not loaded or 'quantize_if4_cpu' is not available.")
        return ext.quantize_if4_cpu(input_tensor, block_size)
    else:
        raise RuntimeError(
            f"No suitable IF4 quantization kernel found for device '{device.type}'. "
            "Please ensure CUDA/Triton extensions are built and device is supported."
        )


def dequantize_from_if4_kernel(
    packed_data: torch.Tensor,
    scales: torch.Tensor,
    block_metadata: torch.Tensor, # This is the encoded_scales tensor
    output_dtype: torch.dtype = torch.float32,
    block_size: int = IF4_BLOCK_SIZE
) -> torch.Tensor:
    """
    High-level dispatcher for dequantizing IF4 packed data to a float32 tensor.
    """
    if not (packed_data.dtype == torch.uint8 and scales.is_floating_point() and block_metadata.is_floating_point()):
        raise TypeError("Input types mismatch for dequantization. Expected uint8 for packed_data, float for scales/metadata.")

    ext = _get_ext_module()
    device = packed_data.device

    if device.type == 'cuda' and _has_triton():
        try:
            from quantizeflow._C.triton import if4_quant as triton_if4_quant
            return triton_if4_quant.dequantize_if4_triton(
                packed_data, scales, block_metadata, output_dtype, block_size
            )
        except Exception as e:
            warnings.warn(f"Triton IF4 dequantization failed: {e}. Falling back to CUDA/CPU.")
            pass

    if device.type == 'cuda' and _has_cuda():
        if ext is None or not hasattr(ext, 'dequantize_if4_cuda'):
            raise RuntimeError("CUDA extensions are not loaded or 'dequantize_if4_cuda' is not available.")
        return ext.dequantize_if4_cuda(packed_data, scales, block_metadata, output_dtype, block_size)
    elif device.type == 'cpu':
        if ext is None or not hasattr(ext, 'dequantize_if4_cpu'):
            raise RuntimeError("CPU extensions are not loaded or 'dequantize_if4_cpu' is not available.")
        return ext.dequantize_if4_cpu(packed_data, scales, block_metadata, output_dtype, block_size)
    else:
        raise RuntimeError(
            f"No suitable IF4 dequantization kernel found for device '{device.type}'. "
            "Please ensure CUDA/Triton extensions are built and device is supported."
        )

# --- Other potential utilities (e.g., for block reshaping, if needed) ---

def reshape_to_blocks(tensor: torch.Tensor, block_size: int = IF4_BLOCK_SIZE) -> torch.Tensor:
    """
    Reshapes a tensor into a view where the last dimension represents blocks.
    This is often a preparatory step for block-wise operations.
    Assumes the last dimension is the one to be blocked.
    """
    if tensor.shape[-1] % block_size != 0:
        raise ValueError(f"Last dimension ({tensor.shape[-1]}) must be divisible by block_size ({block_size}).")

    original_shape = tensor.shape
    new_shape = (*original_shape[:-1], original_shape[-1] // block_size, block_size)
    return tensor.view(new_shape)

def reshape_from_blocks(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reshapes a block-viewed tensor back to its original flat last dimension.
    Assumes the last two dimensions were originally one dimension split into blocks.
    """
    if tensor.dim() < 2:
        raise ValueError("Tensor must have at least two dimensions for reshaping from blocks.")

    original_last_dim = tensor.shape[-2] * tensor.shape[-1]
    original_shape = (*tensor.shape[:-2], original_last_dim)
    return tensor.view(original_shape)

```