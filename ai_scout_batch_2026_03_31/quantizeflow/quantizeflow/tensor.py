```python
import torch
import math

# Constants - These define the block structure and packing details.
# In a real library, these might be configurable or part of a global config.
BLOCK_SIZE = 16  # Number of float values per block in the last dimension
PACKED_ELEMENTS_PER_BYTE = 2 # Two 4-bit values fit into one uint8 byte
BITS_PER_ELEMENT = 4 # Each quantized value is 4 bits (e.g., IF4 means 4 bits per element)

class AdaptiveQuantizedTensor:
    """
    A custom tensor type for dynamic, block-adaptive quantization schemes like IF4.
    It encapsulates packed 4-bit data, per-block scale factors, and metadata
    indicating whether a block is FP4 or INT4.

    The internal representation stores:
    - _packed_data: A torch.uint8 tensor where two 4-bit quantized values are packed into each byte.
                    Its shape is `original_shape[:-1] + (original_shape[-1] // PACKED_ELEMENTS_PER_BYTE,)`.
    - _scales: A floating-point tensor (e.g., torch.float16 or torch.float32) holding the
               per-block scale factors. Its shape is `original_shape[:-1] + (original_shape[-1] // BLOCK_SIZE,)`.
    - _block_types: A boolean tensor (torch.bool) indicating the quantization type for each block.
                    `True` typically means FP4, `False` means INT4. Its shape matches `_scales`.
                    Low-level kernels will likely encode this into the sign bit of the scale factor.
    - _original_shape: The shape of the tensor before quantization.
    - _original_dtype: The dtype of the tensor before quantization (e.g., torch.float32).
    """

    def __init__(self,
                 packed_data: torch.Tensor,
                 scales: torch.Tensor,
                 block_types: torch.Tensor,
                 original_shape: tuple,
                 original_dtype: torch.dtype):
        """
        Initializes an AdaptiveQuantizedTensor.

        Args:
            packed_data (torch.Tensor): A uint8 tensor containing the packed 4-bit quantized values.
                                        Expected shape: `original_shape[:-1] + (original_shape[-1] // PACKED_ELEMENTS_PER_BYTE,)`.
            scales (torch.Tensor): A floating-point tensor containing per-block scale factors.
                                   Expected shape: `original_shape[:-1] + (original_shape[-1] // BLOCK_SIZE,)`.
            block_types (torch.Tensor): A boolean tensor indicating the type of each block (True for FP4, False for INT4).
                                        Its shape must match `scales`.
            original_shape (tuple): The shape of the tensor before quantization.
            original_dtype (torch.dtype): The dtype of the tensor before quantization (e.g., torch.float32).
        """
        # --- Input Validation ---
        if not isinstance(packed_data, torch.Tensor) or packed_data.dtype != torch.uint8:
            raise TypeError("`packed_data` must be a `torch.uint8` tensor.")
        if not isinstance(scales, torch.Tensor) or not scales.is_floating_point():
            raise TypeError("`scales` must be a floating-point `torch.Tensor`.")
        if not isinstance(block_types, torch.Tensor) or block_types.dtype != torch.bool:
            raise TypeError("`block_types` must be a `torch.bool` tensor.")
        if not isinstance(original_shape, tuple) or not all(isinstance(d, int) and d > 0 for d in original_shape):
            raise ValueError("`original_shape` must be a tuple of positive integers.")
        if not isinstance(original_dtype, torch.dtype) or original_dtype not in [torch.float16, torch.float32, torch.bfloat16]:
            raise ValueError(f"Unsupported `original_dtype`: {original_dtype}. Expected `torch.float16`, `torch.float32`, or `torch.bfloat16`.")

        if scales.shape != block_types.shape:
            raise ValueError(f"`scales` and `block_types` must have the same shape. Got {scales.shape} and {block_types.shape}.")

        # Validate tensor shapes based on original_shape and block constants
        if original_shape[-1] % BLOCK_SIZE != 0:
            raise ValueError(f"Last dimension ({original_shape[-1]}) of `original_shape` must be a multiple of "
                             f"BLOCK_SIZE ({BLOCK_SIZE}) for current block strategy.")
        
        expected_packed_shape = original_shape[:-1] + (original_shape[-1] // PACKED_ELEMENTS_PER_BYTE,)
        if packed_data.shape != expected_packed_shape:
             raise ValueError(f"`packed_data` shape {packed_data.shape} does not match expected shape {expected_packed_shape} "
                              f"derived from `original_shape` {original_shape}.")
        
        expected_scales_shape = original_shape[:-1] + (original_shape[-1] // BLOCK_SIZE,)
        if scales.shape != expected_scales_shape:
            raise ValueError(f"`scales` shape {scales.shape} does not match expected shape {expected_scales_shape} "
                             f"derived from `original_shape` {original_shape}.")

        self._packed_data = packed_data
        self._scales = scales
        self._block_types = block_types
        self._original_shape = original_shape
        self._original_dtype = original_dtype
        self._device = packed_data.device


    # --- Properties for introspection ---
    @property
    def packed_data(self) -> torch.Tensor:
        """Returns the packed 4-bit data tensor (uint8)."""
        return self._packed_data

    @property
    def scales(self) -> torch.Tensor:
        """Returns the per-block scale factors tensor (float)."""
        return self._scales

    @property
    def block_types(self) -> torch.Tensor:
        """Returns the boolean tensor indicating block types (True for FP4, False for INT4)."""
        return self._block_types

    @property
    def shape(self) -> tuple:
        """Returns the original shape of the tensor before quantization."""
        return self._original_shape

    @property
    def dtype(self) -> torch.dtype:
        """Returns the original dtype of the tensor before quantization."""
        return self._original_dtype

    @property
    def device(self) -> torch.device:
        """Returns the device of the underlying tensors."""
        return self._device

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the original tensor."""
        return len(self._original_shape)

    def size(self, dim: int = None) -> int:
        """
        Returns the size of a specific dimension, or the full shape if `dim` is None.
        Mimics `torch.Tensor.size()`.
        """
        if dim is None:
            return self._original_shape
        if not isinstance(dim, int):
            raise TypeError("`dim` must be an integer.")
        if dim < -self.ndim or dim >= self.ndim:
            raise IndexError(f"Dimension out of range (expected to be in the range [{-self.ndim}, {self.ndim-1}], but got {dim})")
        return self._original_shape[dim]

    def is_cuda(self) -> bool:
        """Returns True if the tensor resides on a CUDA device."""
        return self._device.type == 'cuda'

    # --- Device transfer methods ---
    def to(self, device: torch.device) -> 'AdaptiveQuantizedTensor':
        """
        Moves all internal tensors of the AdaptiveQuantizedTensor to the specified device.
        """
        if self._device == device:
            return self

        new_packed_data = self._packed_data.to(device)
        new_scales = self._scales.to(device)
        new_block_types = self._block_types.to(device)
        return AdaptiveQuantizedTensor(new_packed_data, new_scales, new_block_types,
                                       self._original_shape, self._original_dtype)

    def cpu(self) -> 'AdaptiveQuantizedTensor':
        """Moves the tensor to CPU memory."""
        return self.to(torch.device('cpu'))

    def cuda(self, device: int = None) -> 'AdaptiveQuantizedTensor':
        """Moves the tensor to CUDA memory."""
        if device is None:
            device = torch.cuda.current_device()
        return self.to(torch.device('cuda', device))

    def __repr__(self):
        """Provides a string representation of the AdaptiveQuantizedTensor."""
        return (f"AdaptiveQuantizedTensor(\n"
                f"  packed_data_shape={self._packed_data.shape}, packed_data_dtype={self._packed_data.dtype},\n"
                f"  scales_shape={self._scales.shape}, scales_dtype={self._scales.dtype},\n"
                f"  block_types_shape={self._block_types.shape}, block_types_dtype={self._block_types.dtype},\n"
                f"  original_shape={self._original_shape}, original_dtype={self._original_dtype},\n"
                f"  device='{self._device}')")


    @classmethod
    def from_float(cls, float_tensor: torch.Tensor) -> 'AdaptiveQuantizedTensor':
        """
        Quantizes a standard float tensor into an AdaptiveQuantizedTensor using IF4 logic.

        NOTE: This implementation is a *simplified, conceptual* Python-only placeholder.
        The actual `QuantizeFlow` library would leverage high-performance C++/CUDA/Triton
        kernels (e.g., from `quantizeflow._C`) for the entire quantization process,
        including:
        1. Lightweight, runtime block analysis (`block_analyzer.cu`).
        2. Efficient 32-bit float to IF4 (packed 4-bit) quantization, including scale factor
           computation and application (`if4_quant.cuh`).
        """
        if not float_tensor.is_floating_point():
            raise TypeError("Input tensor for quantization must be a floating-point tensor.")
        if float_tensor.ndim < 1:
            raise ValueError("Input tensor must have at least one dimension.")
        if float_tensor.shape[-1] % BLOCK_SIZE != 0:
            raise ValueError(f"Last dimension of input tensor ({float_tensor.shape[-1]}) must be a multiple of "
                             f"BLOCK_SIZE ({BLOCK_SIZE}) for block-wise quantization.")

        original_shape = float_tensor.shape
        original_dtype = float_tensor.dtype
        device = float_tensor.device

        # Determine dimensions for blocking (assumes blocking along the last dimension)
        num_blocks_in_last_dim = original_shape[-1] // BLOCK_SIZE
        num_prefix_elements = math.prod(original_shape[:-1]) # Product of all dims except the last one
        num_total_blocks = num_prefix_elements * num_blocks_in_last_dim

        # Prepare temporary tensors to store results before final packing/reshaping.
        # Quantized values are stored as signed 4-bit integers (represented in int8) initially.
        temp_quant_values = torch.empty(num_prefix_elements, original_shape[-1], dtype=torch.int8, device=device)
        temp_scales_flat = torch.empty(num_total_blocks, dtype=original_dtype, device=device)
        temp_block_types_flat = torch.empty(num_total_blocks, dtype=torch.bool, device=device) # True for FP4, False for INT4

        # Reshape the input tensor to `(num_prefix_elements, original_shape[-1])` to simplify block iteration.
        flat_tensor = float_tensor.reshape(num_prefix_elements, original_shape[-1])

        # --- Conceptual Block-wise Quantization Loop (Replaced by Kernels) ---
        for i in range(num_prefix_elements):
            for j in range(num_blocks_in_last_dim):
                block_start_idx = j * BLOCK_SIZE
                block_end_idx = block_start_idx + BLOCK_SIZE
                block = flat_tensor[i, block_start_idx : block_end_idx]

                block_idx = i * num_blocks_in_last_dim + j

                # --- Simplified Block Analysis (Placeholder for `block_analyzer.cu`) ---
                # A heuristic for demonstration: if a block's values are very small or very large,
                # it might benefit more from FP4's dynamic range. Otherwise, INT4.
                max_abs_val = block.abs().max()
                
                # Treat near-zero blocks as INT4 with a small default scale (or 0)
                if max_abs_val < 1e-6:
                    is_fp4_block = False
                # Example: If the max value is > 1.0 (common for activations/weights), prefer FP4
                # This is a very rough approximation, not actual IF4 analysis.
                else:
                    is_fp4_block = (max_abs_val > 1.0) 

                temp_block_types_flat[block_idx] = is_fp4_block

                # --- Simplified Quantization (Placeholder for `if4_quant.cuh`/`.cu`) ---
                if is_fp4_block:
                    # Conceptual FP4 quantization: Scale to fit 4-bit range.
                    # The actual FP4 format is a specific float representation (e.g., E2M1, E3M0).
                    # Here, we model it as a generic scaled 4-bit integer, and the 'FP4' nature
                    # would be in the low-level kernel's interpretation during operations.
                    # Max representable magnitude for a signed 4-bit integer is 2^(BITS_PER_ELEMENT-1) - 1 (i.e., 7)
                    scale_val = max_abs_val / ( (2**(BITS_PER_ELEMENT - 1)) - 1 )
                    if scale_val == 0: scale_val = torch.tensor(1.0, dtype=original_dtype, device=device) # Avoid division by zero
                    
                    quant_block = (block / scale_val).round().clamp(-(2**(BITS_PER_ELEMENT - 1)), (2**(BITS_PER_ELEMENT - 1)) - 1).to(torch.int8)
                    temp_scales_flat[block_idx] = scale_val
                else: # INT4 block
                    # Scale to fit within signed 4-bit integer range [-8, 7]
                    scale_val = max_abs_val / ( (2**(BITS_PER_ELEMENT - 1)) - 1 ) # Max magnitude is 7
                    if scale_val == 0: scale_val = torch.tensor(1.0, dtype=original_dtype, device=device) # Avoid division by zero

                    quant_block = (block / scale_val).round().clamp(-(2**(BITS_PER_ELEMENT - 1)), (2**(BITS_PER_ELEMENT - 1)) - 1).to(torch.int8)
                    temp_scales_flat[block_idx] = scale_val
                
                temp_quant_values[i, block_start_idx : block_end_idx] = quant_block
        
        # Reshape flat scales and block_types to match `original_shape[:-1] + (num_blocks_in_last_dim,)`
        scales_shape = original_shape[:-1] + (num_blocks_in_last_dim,)
        scales = temp_scales_flat.reshape(scales_shape)
        block_types = temp_block_types_flat.reshape(scales_shape)

        # --- Pack 4-bit values into uint8 bytes ---
        # The `temp_quant_values` are `int8` (e.g., -8 to 7). For packing into `uint8` bytes,
        # we map them to an unsigned range (e.g., 0 to 15) by adding an offset.
        # This mapping `value + 2^(BITS_PER_ELEMENT-1)` is a common approach.
        shifted_quant_values = (temp_quant_values + (2**(BITS_PER_ELEMENT - 1))).to(torch.uint8) # Values now 0-15

        # Reshape `shifted_quant_values` to group elements for packing:
        # From (num_prefix_elements, original_shape[-1])
        # To (num_prefix_elements, original_shape[-1] // PACKED_ELEMENTS_PER_BYTE, PACKED_ELEMENTS_PER_BYTE)
        values_to_pack = shifted_quant_values.reshape(
            num_prefix_elements,
            original_shape[-1] // PACKED_ELEMENTS_PER_BYTE,
            PACKED_ELEMENTS_PER_BYTE
        )

        val1 = values_to_pack[..., 0] # Represents the high 4 bits for each byte
        val2 = values_to_pack[..., 1] # Represents the low 4 bits for each byte

        # Combine two 4-bit values into one 8-bit byte
        packed_data_interim = (val1 << BITS_PER_ELEMENT) | val2

        # Reshape to the final `packed_data` shape:
        # `original_shape[:-1] + (original_shape[-1] // PACKED_ELEMENTS_PER_BYTE,)`
        packed_data_shape = original_shape[:-1] + (original_shape[-1] // PACKED_ELEMENTS_PER_BYTE,)
        packed_data = packed_data_interim.reshape(packed_data_shape)

        return cls(packed_data, scales, block_types, original_shape, original_dtype)


    def dequantize(self) -> torch.Tensor:
        """
        Dequantizes the AdaptiveQuantizedTensor back to a standard float tensor.

        NOTE: This implementation is a *simplified, conceptual* Python-only placeholder.
        The actual `QuantizeFlow` library would leverage high-performance C++/CUDA/Triton
        kernels (e.g., from `quantizeflow._C`) for efficient dequantization.
        """
        device = self._device
        original_shape = self._original_shape
        original_dtype = self._original_dtype

        num_blocks_in_last_dim = original_shape[-1] // BLOCK_SIZE
        num_prefix_elements = math.prod(original_shape[:-1])

        # --- Unpack 4-bit values from uint8 bytes ---
        # Reshape `_packed_data` to iterate through bytes:
        # From `original_shape[:-1] + (N_packed_dim,)` to `(num_prefix_elements, N_packed_dim)`
        packed_data_flat = self._packed_data.reshape(num_prefix_elements, self._packed_data.shape[-1])

        # Extract two 4-bit values from each byte.
        # This results in two tensors, each (num_prefix_elements, N_packed_dim).
        val1_uint8 = (packed_data_flat >> BITS_PER_ELEMENT) & 0xF # High 4 bits
        val2_uint8 = packed_data_flat & 0xF                       # Low 4 bits

        # Stack them and reshape to get the interleaved sequence of all 4-bit values:
        # `torch.stack([val1, val2], dim=-1)` creates (N, M, 2)
        # `.reshape(num_prefix_elements, original_shape[-1])` flattens the last two dims.
        unpacked_values_flat_uint8 = torch.stack([val1_uint8, val2_uint8], dim=-1).reshape(num_prefix_elements, original_shape[-1])

        # Convert back from unsigned [0, 15] to signed 4-bit (e.g., -8 to 7).
        # This reverses the `value + 2^(BITS_PER_ELEMENT-1)` mapping used during packing.
        unpacked_values_flat_int8 = (unpacked_values_flat_uint8 - (2**(BITS_PER_ELEMENT - 1))).to(torch.int8)

        dequantized_tensor_flat = torch.empty_like(unpacked_values_flat_int8, dtype=original_dtype)

        # Reshape scales and block_types to match the iteration logic (flattened)
        scales_flat = self._scales.reshape(-1)
        block_types_flat = self._block_types.reshape(-1) # Note: block_types not strictly needed for symmetric dequantization,
                                                         # but kept for completeness in this conceptual example.

        # --- Conceptual Block-wise Dequantization Loop (Replaced by Kernels) ---
        for i in range(num_prefix_elements):
            for j in range(num_blocks_in_last_dim):
                block_start_idx = j * BLOCK_SIZE
                block_end_idx = block_start_idx + BLOCK_SIZE
                
                block_idx = i * num_blocks_in_last_dim + j

                quant_block = unpacked_values_flat_int8[i, block_start_idx : block_end_idx]
                scale = scales_flat[block_idx]
                # is_fp4_block = block_types_flat[block_idx] # Not used in simple symmetric dequant.

                # Apply scale to dequantize
                dequantized_block = quant_block.to(original_dtype) * scale
                dequantized_tensor_flat[i, block_start_idx : block_end_idx] = dequantized_block
        
        return dequantized_tensor_flat.reshape(original_shape)

```