```python
import abc
from typing import List, Tuple, Dict, Any

class BaseAccelerator(abc.ABC):
    """
    Abstract base class for specialized AI accelerators.

    This class defines the interface for different hardware models, allowing
    for 'plug-and-play' experimentation with various hardware paradigms
    and their impact on LLM inference. Concrete accelerator models (e.g.,
    Photonic, Analog, Neuromorphic) will inherit from this class and
    implement its abstract methods to provide their specific simulation logic.
    """

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Returns the unique name of this accelerator model.

        Returns:
            str: The name of the accelerator (e.g., "PhotonicAI", "AnalogMLP", "SpikeCore").
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_architecture_description(self) -> Dict[str, Any]:
        """
        Provides a detailed description of the accelerator's architecture.

        This dictionary can include information about:
        - 'compute_units': (e.g., {'MACs_per_cycle': 4096, 'frequency_ghz': 2.0})
        - 'memory_hierarchy': (e.g., {'HBM': {'capacity_gb': 64, 'bandwidth_gbps': 1024}, 'L1_cache_kb': 256})
        - 'special_features': (e.g., {'sparse_acceleration': True, 'quantization_units': ['INT8', 'FP8']})

        Returns:
            Dict[str, Any]: A dictionary describing the architectural features.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def simulate_latency(self, operation_type: str, operand_shapes: List[Tuple[int, ...]],
                         dtype: str, batch_size: int = 1, **kwargs) -> float:
        """
        Simulates the computational latency for a given operation on the accelerator.

        Args:
            operation_type (str): Type of operation (e.g., "matmul", "conv", "add", "attention_block", "softmax").
            operand_shapes (List[Tuple[int, ...]]): A list of tuples, where each tuple
                                                    represents the shape of an input operand.
                                                    Example: for matmul (A @ B), it could be `[(M, K), (K, N)]`.
            dtype (str): Data type of the computation (e.g., "FP32", "FP16", "BF16", "INT8", "FP8").
            batch_size (int): The batch size for the operation. Defaults to 1.
            **kwargs: Additional parameters specific to the operation or hardware simulation
                      (e.g., 'sparsity_ratio', 'attention_head_count').

        Returns:
            float: Estimated latency in seconds.

        Raises:
            ValueError: If the `dtype` or `operation_type` is not supported.
        """
        self._validate_dtype(dtype)
        # Concrete implementations should also validate operation_type
        raise NotImplementedError

    @abc.abstractmethod
    def simulate_energy(self, operation_type: str, operand_shapes: List[Tuple[int, ...]],
                        dtype: str, batch_size: int = 1, **kwargs) -> float:
        """
        Simulates the estimated energy consumption for a given operation on the accelerator.

        Args:
            operation_type (str): Type of operation (e.g., "matmul", "conv", "add", "attention_block", "softmax").
            operand_shapes (List[Tuple[int, ...]]): A list of tuples, where each tuple
                                                    represents the shape of an input operand.
            dtype (str): Data type of the computation (e.g., "FP32", "FP16", "BF16", "INT8", "FP8").
            batch_size (int): The batch size for the operation. Defaults to 1.
            **kwargs: Additional parameters specific to the operation or hardware simulation.

        Returns:
            float: Estimated energy consumption in Joules.

        Raises:
            ValueError: If the `dtype` or `operation_type` is not supported.
        """
        self._validate_dtype(dtype)
        # Concrete implementations should also validate operation_type
        raise NotImplementedError

    @abc.abstractmethod
    def simulate_throughput(self, operation_type: str, operand_shapes: List[Tuple[int, ...]],
                            dtype: str, batch_size: int = 1, **kwargs) -> float:
        """
        Simulates the estimated throughput for a given operation on the accelerator.

        Throughput can be operations per second (e.g., FLOPs/s, MACs/s) or inferences per second.
        The specific unit should be defined by the concrete implementation.

        Args:
            operation_type (str): Type of operation (e.g., "matmul", "conv", "add", "attention_block", "softmax").
            operand_shapes (List[Tuple[int, ...]]): A list of tuples, where each tuple
                                                    represents the shape of an input operand.
            dtype (str): Data type of the computation (e.g., "FP32", "FP16", "BF16", "INT8", "FP8").
            batch_size (int): The batch size for the operation. Defaults to 1.
            **kwargs: Additional parameters specific to the operation or hardware simulation.

        Returns:
            float: Estimated throughput in operations per second (or relevant unit).

        Raises:
            ValueError: If the `dtype` or `operation_type` is not supported.
        """
        self._validate_dtype(dtype)
        # Concrete implementations should also validate operation_type
        raise NotImplementedError

    @abc.abstractmethod
    def simulate_memory_access_latency(self, access_type: str, data_size_bytes: int,
                                       memory_level: str = "main_memory", **kwargs) -> float:
        """
        Simulates the latency for a memory access operation (read, write, or transfer).

        Args:
            access_type (str): Type of memory access (e.g., "read", "write", "transfer").
            data_size_bytes (int): The size of the data being accessed in bytes.
            memory_level (str): The level of memory being accessed (e.g., "L1_cache", "HBM", "DRAM", "on_chip_memory").
            **kwargs: Additional parameters specific to the memory access (e.g., 'is_sequential', 'cache_hit_ratio').

        Returns:
            float: Estimated memory access latency in seconds.

        Raises:
            ValueError: If the `access_type` or `memory_level` is not supported.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def simulate_memory_access_energy(self, access_type: str, data_size_bytes: int,
                                      memory_level: str = "main_memory", **kwargs) -> float:
        """
        Simulates the energy consumption for a memory access operation.

        Args:
            access_type (str): Type of memory access (e.g., "read", "write", "transfer").
            data_size_bytes (int): The size of the data being accessed in bytes.
            memory_level (str): The level of memory being accessed (e.g., "L1_cache", "HBM", "DRAM", "on_chip_memory").
            **kwargs: Additional parameters specific to the memory access.

        Returns:
            float: Estimated memory access energy in Joules.

        Raises:
            ValueError: If the `access_type` or `memory_level` is not supported.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_supported_datatypes(self) -> List[str]:
        """
        Returns a list of data types supported by the accelerator.

        Example: ["FP32", "FP16", "BF16", "INT8", "FP8"]

        Returns:
            List[str]: A list of supported data type strings.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_supported_quantization_schemes(self) -> List[str]:
        """
        Returns a list of quantization schemes natively supported or efficiently
        accelerated by the hardware.

        Example: ["per_tensor_affine", "per_channel_affine", "block_fp_quantization", "stochastic_rounding"]

        Returns:
            List[str]: A list of supported quantization scheme strings.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def supports_sparse_operations(self) -> bool:
        """
        Indicates whether the accelerator has specialized hardware or features
        to efficiently handle sparse tensor operations (e.g., sparse matrix multiplication).

        Returns:
            bool: True if sparse operations are efficiently supported, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def supports_dynamic_batching(self) -> bool:
        """
        Indicates whether the accelerator efficiently supports dynamic batch sizes
        without significant performance penalties due to re-compilation, padding, etc.

        Returns:
            bool: True if dynamic batching is efficiently supported, False otherwise.
        """
        raise NotImplementedError

    def _validate_dtype(self, dtype: str):
        """
        Internal helper to validate if a given data type is supported by this accelerator.

        Args:
            dtype (str): The data type to validate.

        Raises:
            ValueError: If the data type is not supported.
        """
        if dtype not in self.get_supported_datatypes():
            raise ValueError(
                f"Data type '{dtype}' is not supported by {self.get_name()}. "
                f"Supported types: {self.get_supported_datatypes()}"
            )
```