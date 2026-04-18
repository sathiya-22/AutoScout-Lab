```python
from abc import ABC, abstractmethod
import math

# Define a base abstract class for hardware models to ensure modularity
class HardwareModel(ABC):
    """
    Abstract base class for all simulated hardware models.
    Defines the interface for hardware capabilities and operation simulation.
    """
    @abstractmethod
    def simulate_operation(self, op_type: str, **kwargs) -> tuple[float, float]:
        """
        Simulates the latency and energy consumption of a specific operation
        on this hardware model.

        Args:
            op_type (str): The type of operation to simulate (e.g., 'matrix_multiply',
                           'memory_read', 'activation_function').
            **kwargs: Operation-specific parameters (e.g., dimensions, data size, precision_bits).

        Returns:
            tuple[float, float]: A tuple containing (latency_us, energy_joules).
        """
        pass

    @abstractmethod
    def supports_quantization(self, precision_bits: int) -> bool:
        """
        Checks if the hardware model supports a given quantization precision.

        Args:
            precision_bits (int): The number of bits for quantization (e.g., 8 for INT8).

        Returns:
            bool: True if supported, False otherwise.
        """
        pass

    @abstractmethod
    def supports_sparse_computation(self) -> bool:
        """
        Indicates if the hardware model has specific optimizations for sparse computations.

        Returns:
            bool: True if sparse computation is specifically supported/optimized, False otherwise.
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> dict:
        """
        Returns a dictionary detailing the capabilities and parameters of the hardware model.
        """
        pass


class AnalogModel(HardwareModel):
    """
    Simulated Analog AI accelerator model.

    This model provides estimates for latency and energy consumption for common
    LLM operations, reflecting characteristics typical of analog computing paradigms
    such as in-memory computation, inherent low-precision, and energy efficiency
    for multiply-accumulate operations.
    """

    # --- Default Characteristic Parameters for a Hypothetical Analog Chip ---
    # These values are illustrative and can be tuned for different analog architectures.

    # Energy and Latency per effective Multiply-Accumulate (MAC) operation
    # Analog often boasts very low energy per MAC.
    BASE_MAC_ENERGY_PJ = 0.05  # Picojoules per MAC (e.g., 50 fJ)
    BASE_MAC_LATENCY_NS = 0.5  # Nanoseconds per MAC equivalent (e.g., 0.5 ns per effective cycle)

    # Memory Access Costs (assuming some on-chip or near-chip memory interface)
    MEMORY_ACCESS_ENERGY_PJ_PER_BYTE = 5.0  # PJ per byte for data movement
    MEMORY_ACCESS_LATENCY_NS_PER_BYTE = 1.0  # NS per byte for data movement

    # ADC/DAC conversion overheads (per bit, per value)
    # Relevant when interfacing with digital memory or host, or within the analog domain itself
    # if intermediate digital steps are required.
    ADC_DAC_ENERGY_PJ_PER_BIT_PER_VALUE = 0.1  # PJ per bit for conversion
    ADC_DAC_LATENCY_NS_PER_BIT_PER_VALUE = 0.1  # NS per bit for conversion

    # Max and Min precision supported by the analog circuitry (effective bits of resolution)
    MAX_SUPPORTED_PRECISION_BITS = 8  # Typically 4-8 bits for current analog designs
    MIN_SUPPORTED_PRECISION_BITS = 2  # Analog often requires at least a few bits for signal integrity

    # Overhead factors for non-linear activations (e.g., ReLU, GeLU implementation in analog)
    ACTIVATION_ENERGY_FACTOR = 1.5  # Multiplier for base MAC energy for activations
    ACTIVATION_LATENCY_FACTOR = 1.2  # Multiplier for base MAC latency for activations

    # Sparse computation support - Analog can potentially be designed to exploit sparsity
    # but naive implementations might not directly benefit as much as digital sparse engines.
    SUPPORTS_SPARSE = False  # Default to False, specific designs could enable this.
    # If supports_sparse is True, this factor reduces the cost of sparse operations
    SPARSE_EFFICIENCY_FACTOR = 0.8 # e.g., 0.8 means 20% energy/latency reduction for sparse ops

    def __init__(self,
                 mac_energy_pj: float = BASE_MAC_ENERGY_PJ,
                 mac_latency_ns: float = BASE_MAC_LATENCY_NS,
                 mem_energy_pj_per_byte: float = MEMORY_ACCESS_ENERGY_PJ_PER_BYTE,
                 mem_latency_ns_per_byte: float = MEMORY_ACCESS_LATENCY_NS_PER_BYTE,
                 adc_dac_energy_pj_per_bit: float = ADC_DAC_ENERGY_PJ_PER_BIT_PER_VALUE,
                 adc_dac_latency_ns_per_bit: float = ADC_DAC_LATENCY_NS_PER_BIT_PER_VALUE,
                 max_precision_bits: int = MAX_SUPPORTED_PRECISION_BITS,
                 min_precision_bits: int = MIN_SUPPORTED_PRECISION_BITS,
                 supports_sparse: bool = SUPPORTS_SPARSE,
                 sparse_efficiency_factor: float = SPARSE_EFFICIENCY_FACTOR):
        """
        Initializes the AnalogModel with specific hardware characteristics.

        Args:
            mac_energy_pj (float): Energy cost per MAC operation in picojoules.
            mac_latency_ns (float): Latency per MAC operation in nanoseconds.
            mem_energy_pj_per_byte (float): Energy cost per byte of memory access in picojoules.
            mem_latency_ns_per_byte (float): Latency per byte of memory access in nanoseconds.
            adc_dac_energy_pj_per_bit (float): Energy cost per bit per value for ADC/DAC conversion in picojoules.
            adc_dac_latency_ns_per_bit (float): Latency cost per bit per value for ADC/DAC conversion in nanoseconds.
            max_precision_bits (int): Maximum bit precision supported for computation.
            min_precision_bits (int): Minimum bit precision supported for computation.
            supports_sparse (bool): True if the model has mechanisms to optimize sparse computations.
            sparse_efficiency_factor (float): Factor to multiply costs by for sparse ops (0.0 to 1.0).
        """
        if not (0 < min_precision_bits <= max_precision_bits):
            raise ValueError("Invalid precision bit range: min_precision_bits must be <= max_precision_bits and > 0.")
        if not (0.0 <= sparse_efficiency_factor <= 1.0):
            raise ValueError("Sparse efficiency factor must be between 0.0 and 1.0 (inclusive).")
        if not all(val >= 0 for val in [mac_energy_pj, mac_latency_ns, mem_energy_pj_per_byte,
                                        mem_latency_ns_per_byte, adc_dac_energy_pj_per_bit,
                                        adc_dac_latency_ns_per_bit]):
            raise ValueError("All energy and latency parameters must be non-negative.")

        self._mac_energy_pj = mac_energy_pj
        self._mac_latency_ns = mac_latency_ns
        self._mem_energy_pj_per_byte = mem_energy_pj_per_byte
        self._mem_latency_ns_per_byte = mem_latency_ns_per_byte
        self._adc_dac_energy_pj_per_bit = adc_dac_energy_pj_per_bit
        self._adc_dac_latency_ns_per_bit = adc_dac_latency_ns_per_bit
        self._max_precision_bits = max_precision_bits
        self._min_precision_bits = min_precision_bits
        self._supports_sparse = supports_sparse
        self._sparse_efficiency_factor = sparse_efficiency_factor

        # Store capabilities for easy access
        self._capabilities = {
            "type": "Analog AI Accelerator",
            "mac_energy_pj": self._mac_energy_pj,
            "mac_latency_ns": self._mac_latency_ns,
            "memory_energy_pj_per_byte": self._mem_energy_pj_per_byte,
            "memory_latency_ns_per_byte": self._mem_latency_ns_per_byte,
            "adc_dac_energy_pj_per_bit_per_value": self._adc_dac_energy_pj_per_bit,
            "adc_dac_latency_ns_per_bit_per_value": self._adc_dac_latency_ns_per_bit,
            "max_supported_precision_bits": self._max_precision_bits,
            "min_supported_precision_bits": self._min_precision_bits,
            "supports_sparse_computation": self._supports_sparse,
            "sparse_efficiency_factor": self._sparse_efficiency_factor,
        }

    def _calculate_adc_dac_cost(self, num_values: int, precision_bits: int) -> tuple[float, float]:
        """Calculates energy and latency for ADC/DAC conversions."""
        if not self.supports_quantization(precision_bits):
            # If precision is outside supported range, conversion might be impossible or incur extreme costs.
            # For simplicity, returning zero assumes caller handles this or it's implicitly part of larger errors.
            return 0.0, 0.0
        
        energy_pj = num_values * precision_bits * self._adc_dac_energy_pj_per_bit
        latency_ns = num_values * precision_bits * self._adc_dac_latency_ns_per_bit
        return energy_pj, latency_ns

    def _calculate_mac_cost(self, num_mac_ops: int, precision_bits: int) -> tuple[float, float]:
        """
        Calculates energy and latency for Multiply-Accumulate operations.
        Assumes higher precision might incur slight overheads or lower throughput
        due to analog design constraints (e.g., more complex circuitry, longer integration times).
        """
        if not self.supports_quantization(precision_bits):
            raise ValueError(f"Precision {precision_bits} bits is not supported by this AnalogModel "
                             f"(min: {self._min_precision_bits}, max: {self._max_precision_bits}).")

        # Simple model: assume the base MAC cost already implicitly reflects the design for max_precision.
        # Additional precision penalty could be modeled here if needed (e.g., by higher factor for higher bits).
        energy_pj = num_mac_ops * self._mac_energy_pj
        latency_ns = num_mac_ops * self._mac_latency_ns

        # Optional: Add a precision penalty if higher precision costs more than linearly
        # precision_scale = 1.0 + (precision_bits - self._min_precision_bits) * 0.02 # 2% penalty per bit
        # energy_pj *= precision_scale
        # latency_ns *= precision_scale

        return energy_pj, latency_ns

    def simulate_operation(self, op_type: str, **kwargs) -> tuple[float, float]:
        """
        Simulates the latency and energy consumption of a specific operation.

        Args:
            op_type (str): The type of operation. Supported types:
                           'matrix_multiply', 'vector_add', 'elementwise_multiply',
                           'activation_function', 'memory_read', 'memory_write'.
            **kwargs: Parameters specific to the operation type.
                For 'matrix_multiply': M, N, K, precision_bits, is_sparse (bool), sparsity_ratio (float).
                For 'vector_add', 'elementwise_multiply', 'activation_function': num_elements, precision_bits.
                For 'memory_read', 'memory_write': data_size_bytes.

        Returns:
            tuple[float, float]: A tuple containing (latency_us, energy_joules).
        """
        total_latency_ns = 0.0
        total_energy_pj = 0.0

        op_type = op_type.lower()

        if op_type == 'matrix_multiply':
            M = kwargs.get('M')
            N = kwargs.get('N')
            K = kwargs.get('K')
            precision_bits = kwargs.get('precision_bits')
            is_sparse = kwargs.get('is_sparse', False)
            sparsity_ratio = kwargs.get('sparsity_ratio', 0.0)  # 0.0 means dense, 1.0 means fully sparse

            if not all(isinstance(arg, int) and arg > 0 for arg in [M, N, K, precision_bits]):
                raise ValueError(f"Invalid dimensions or precision for matrix_multiply: M={M}, N={N}, K={K}, precision_bits={precision_bits}. All must be positive integers.")

            if not self.supports_quantization(precision_bits):
                raise ValueError(f"Matrix Multiply with {precision_bits}-bit precision not supported by this AnalogModel.")

            # Number of MAC operations: M * N * K
            num_mac_ops = M * N * K

            # Calculate MAC cost
            mac_energy_pj, mac_latency_ns = self._calculate_mac_cost(num_mac_ops, precision_bits)
            total_energy_pj += mac_energy_pj
            total_latency_ns += mac_latency_ns

            # ADC/DAC conversion cost for input/output data.
            # Assume inputs A (M, K), B (K, N) and output C (M, N) require conversion.
            num_input_values = M * K + K * N
            num_output_values = M * N

            input_adc_dac_energy, input_adc_dac_latency = self._calculate_adc_dac_cost(num_input_values, precision_bits)
            output_adc_dac_energy, output_adc_dac_latency = self._calculate_adc_dac_cost(num_output_values, precision_bits)

            total_energy_pj += input_adc_dac_energy + output_adc_dac_energy
            total_latency_ns += input_adc_dac_latency + output_adc_dac_latency

            # Apply sparse efficiency if supported and operation is sparse
            if is_sparse and self._supports_sparse and sparsity_ratio > 0.0 and sparsity_ratio < 1.0:
                # A simplified model: assume a flat efficiency gain for sparse operations.
                # In reality, this is complex and depends on sparse data representation/hardware.
                total_energy_pj *= self._sparse_efficiency_factor
                total_latency_ns *= self._sparse_efficiency_factor

        elif op_type in ['vector_add', 'elementwise_multiply']:
            num_elements = kwargs.get('num_elements')
            precision_bits = kwargs.get('precision_bits')

            if not all(isinstance(arg, int) and arg > 0 for arg in [num_elements, precision_bits]):
                raise ValueError(f"Invalid arguments for {op_type}: num_elements={num_elements}, precision_bits={precision_bits}. Both must be positive integers.")

            if not self.supports_quantization(precision_bits):
                raise ValueError(f"{op_type} with {precision_bits}-bit precision not supported by this AnalogModel.")

            # For element-wise operations, assume 1 effective MAC-like op per element
            num_mac_like_ops = num_elements
            mac_energy_pj, mac_latency_ns = self._calculate_mac_cost(num_mac_like_ops, precision_bits)
            total_energy_pj += mac_energy_pj
            total_latency_ns += mac_latency_ns

            # ADC/DAC for two inputs and one output
            input_adc_dac_energy, input_adc_dac_latency = self._calculate_adc_dac_cost(num_elements * 2, precision_bits)
            output_adc_dac_energy, output_adc_dac_latency = self._calculate_adc_dac_cost(num_elements, precision_bits)

            total_energy_pj += input_adc_dac_energy + output_adc_dac_energy
            total_latency_ns += input_adc_dac_latency + output_adc_dac_latency

        elif op_type == 'activation_function':
            num_elements = kwargs.get('num_elements')
            precision_bits = kwargs.get('precision_bits')
            # activation_type = kwargs.get('activation_type', 'relu') # Could be used for more specific modeling

            if not all(isinstance(arg, int) and arg > 0 for arg in [num_elements, precision_bits]):
                raise ValueError(f"Invalid arguments for activation_function: num_elements={num_elements}, precision_bits={precision_bits}. Both must be positive integers.")

            if not self.supports_quantization(precision_bits):
                raise ValueError(f"Activation function with {precision_bits}-bit precision not supported by this AnalogModel.")

            # Analog implementations of activations can vary. Model it as a factor over element-wise MAC-like operations.
            num_mac_like_ops = num_elements
            base_energy_pj, base_latency_ns = self._calculate_mac_cost(num_mac_like_ops, precision_bits)

            total_energy_pj += base_energy_pj * self.ACTIVATION_ENERGY_FACTOR
            total_latency_ns += base_latency_ns * self.ACTIVATION_LATENCY_FACTOR

            # ADC/DAC for one input and one output
            input_adc_dac_energy, input_adc_dac_latency = self._calculate_adc_dac_cost(num_elements, precision_bits)
            output_adc_dac_energy, output_adc_dac_latency = self._calculate_adc_dac_cost(num_elements, precision_bits)

            total_energy_pj += input_adc_dac_energy + output_adc_dac_energy
            total_latency_ns += input_adc_dac_latency + output_adc_dac_latency

        elif op_type in ['memory_read', 'memory_write']:
            data_size_bytes = kwargs.get('data_size_bytes')

            if not isinstance(data_size_bytes, (int, float)) or data_size_bytes < 0:
                raise ValueError(f"Invalid data_size_bytes for {op_type}: {data_size_bytes}. Must be non-negative.")

            total_energy_pj += data_size_bytes * self._mem_energy_pj_per_byte
            total_latency_ns += data_size_bytes * self._mem_latency_ns_per_byte

        else:
            raise ValueError(f"Unsupported operation type: {op_type}")

        # Convert picojoules to joules, nanoseconds to microseconds
        latency_us = total_latency_ns / 1000.0  # ns to us
        energy_joules = total_energy_pj / 1e12  # pJ to J

        return latency_us, energy_joules

    def supports_quantization(self, precision_bits: int) -> bool:
        """
        Checks if the analog model supports the given quantization precision.
        Analog models are often optimized for specific (usually lower) precision ranges.
        """
        return self._min_precision_bits <= precision_bits <= self._max_precision_bits

    def supports_sparse_computation(self) -> bool:
        """
        Indicates if the analog model has specific optimizations for sparse computations.
        """
        return self._supports_sparse

    def get_capabilities(self) -> dict:
        """
        Returns a dictionary detailing the capabilities and parameters of the analog model.
        """
        return self._capabilities

```