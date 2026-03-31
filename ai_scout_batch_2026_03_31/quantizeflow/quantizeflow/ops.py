```python
import torch
from typing import Optional, Union

# Assuming these are defined in their respective files within the quantizeflow package
from .tensor import AdaptiveQuantizedTensor
from .autograd import _QuantizeFlowMatmul, _QuantizeFlowLinear

# Define a common type alias for inputs that can be either float or quantized
InputTensorType = Union[torch.Tensor, AdaptiveQuantizedTensor]

def adaptive_quant_matmul(
    input: InputTensorType,
    weight: AdaptiveQuantizedTensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Performs an adaptive quantized matrix multiplication (GEMM).

    The `weight` tensor must be an `AdaptiveQuantizedTensor`.
    The `input` tensor can be a standard `torch.Tensor` (float32/bfloat16)
    which will be dynamically quantized block-wise based on its distribution,
    or it can already be an `AdaptiveQuantizedTensor`.

    Args:
        input (Union[torch.Tensor, AdaptiveQuantizedTensor]): The left-hand
            side tensor. If a standard float tensor, it will be quantized
            on-the-fly using IF4 logic. If an AdaptiveQuantizedTensor, it is
            used directly.
        weight (AdaptiveQuantizedTensor): The right-hand side tensor,
            expected to be pre-quantized using IF4 logic.
        bias (Optional[torch.Tensor]): An optional bias tensor to add to
            the result. Must be a standard float tensor (e.g., float32).

    Returns:
        torch.Tensor: The dequantized result of the matrix multiplication,
                      as a float32 tensor.

    Raises:
        TypeError: If input or weight types are incorrect.
        ValueError: If tensor shapes are incompatible or devices mismatch.
    """
    if not isinstance(weight, AdaptiveQuantizedTensor):
        raise TypeError(f"Weight must be an AdaptiveQuantizedTensor, but got {type(weight)}")

    if not isinstance(input, (torch.Tensor, AdaptiveQuantizedTensor)):
        raise TypeError(
            f"Input must be a torch.Tensor or AdaptiveQuantizedTensor, "
            f"but got {type(input)}"
        )
    
    # Determine the target device from the quantized weight, as it's the most critical component.
    target_device = weight.device

    # Ensure input is on the same device as weight.
    # AdaptiveQuantizedTensor.to() should handle its internal components correctly.
    if input.device != target_device:
        input = input.to(target_device)

    # If bias is provided, ensure it's a torch.Tensor and on the same device.
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError(f"Bias must be a torch.Tensor or None, but got {type(bias)}")
        if bias.device != target_device:
            bias = bias.to(target_device)
        
        # Basic bias shape validation: Bias typically applies to the last dimension
        # of the output. For a `(..., K) @ (K, N)` -> `(..., N)` matmul,
        # bias should be broadcastable to `(..., N)`, commonly `(N,)`.
        # More robust shape validation will occur in the `_QuantizeFlowMatmul.forward`
        # for complex broadcasting scenarios.
        if bias.ndim > 1 and bias.shape[-1] != weight.shape[-1]:
             raise ValueError(
                 f"Bias shape {bias.shape} incompatible with expected output "
                 f"dimension {weight.shape[-1]} from weight. Expected bias.shape[-1] "
                 f"to match weight.shape[-1] if ndim > 1."
             )
        elif bias.ndim == 1 and bias.shape[0] != weight.shape[-1]:
            raise ValueError(
                f"Bias shape {bias.shape} incompatible with expected output "
                f"dimension {weight.shape[-1]} from weight. Expected bias.shape to be "
                f"({weight.shape[-1]},) if ndim == 1."
            )


    # The actual dispatch to low-level kernels (CUDA/Triton/CPU) and autograd handling
    # is encapsulated within _QuantizeFlowMatmul.apply(). This function will also
    # handle dynamic quantization of `input` if it's a float tensor within its forward pass.
    result = _QuantizeFlowMatmul.apply(input, weight, bias)

    return result


def adaptive_quant_linear(
    input: InputTensorType,
    weight: AdaptiveQuantizedTensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Performs an adaptive quantized linear transformation (fully connected layer).

    This function is equivalent to `input @ weight.T + bias` where `weight`
    is an `AdaptiveQuantizedTensor` and `input` can be dynamically quantized.
    The `weight` is assumed to be in `(out_features, in_features)` layout
    for a standard linear layer.

    Args:
        input (Union[torch.Tensor, AdaptiveQuantizedTensor]): The input tensor.
            Typically of shape `(*, in_features)`. If a standard float tensor,
            it will be quantized on-the-fly using IF4 logic.
        weight (AdaptiveQuantizedTensor): The weight matrix, expected to be
            pre-quantized using IF4 logic, typically of shape
            `(out_features, in_features)`.
        bias (Optional[torch.Tensor]): An optional bias tensor to add to
            the result, typically of shape `(out_features)`. Must be a
            standard float tensor (e.g., float32).

    Returns:
        torch.Tensor: The dequantized result of the linear transformation,
                      as a float32 tensor, typically of shape `(*, out_features)`.

    Raises:
        TypeError: If input or weight types are incorrect.
        ValueError: If tensor shapes are incompatible or devices mismatch.
    """
    if not isinstance(weight, AdaptiveQuantizedTensor):
        raise TypeError(f"Weight must be an AdaptiveQuantizedTensor, but got {type(weight)}")

    if not isinstance(input, (torch.Tensor, AdaptiveQuantizedTensor)):
        raise TypeError(
            f"Input must be a torch.Tensor or AdaptiveQuantizedTensor, "
            f"but got {type(input)}"
        )
            
    # Determine the target device from the quantized weight.
    target_device = weight.device

    # Ensure input is on the same device as weight.
    if input.device != target_device:
        input = input.to(target_device)

    # If bias is provided, ensure it's a torch.Tensor and on the same device.
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError(f"Bias must be a torch.Tensor or None, but got {type(bias)}")
        if bias.device != target_device:
            bias = bias.to(target_device)
        
        # Bias for a linear layer is typically 1D and matches the output features.
        # Assuming weight is (out_features, in_features)
        if not (bias.ndim == 1 and bias.shape[0] == weight.shape[0]):
             raise ValueError(
                 f"Bias shape {bias.shape} incompatible with weight's output "
                 f"features {weight.shape[0]}. Expected bias.shape to be ({weight.shape[0]},)."
             )

    # The actual dispatch to low-level kernels and autograd handling
    # is encapsulated within _QuantizeFlowLinear.apply(). This function will
    # handle dynamic quantization of `input` if it's a float tensor, and
    # internally perform the equivalent of `input @ weight.T` with the quantized
    # kernels before adding bias.
    result = _QuantizeFlowLinear.apply(input, weight, bias)

    return result
```