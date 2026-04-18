import math

class QuantHardwareUnit:
    """
    Simulates a specialized hardware unit designed for quantized LLM inference.

    This unit provides methods to estimate computational costs (latency, energy, throughput)
    and memory access patterns for common LLM operations at various precisions.
    It serves as a modular abstraction for different specialized AI accelerator models.
    """

    def __init__(self,
                 unit_id: str,
                 supported_datatypes: list[str],
                 peak_ops_per_datatype: dict[str, float], # e.g., {'INT8': 256e12} (TOPS)
                 energy_per_op_per_datatype: dict[str, float], # e.g., {'INT8': 0.1e-12} (pJ/op)
                 memory_bandwidth_gbps: float, # e.g., 1024 (GB/s)
                 latency_overhead_ns: float = 0.5, # Base latency/pipeline overhead per operation (ns)
                 num_cores: int = 128,
                 name: str = "GenericQuantUnit"):
        """
        Initializes the simulated quantized hardware unit.

        Args:
            unit_id (str): A unique identifier for this hardware unit instance.
            supported_datatypes (list[str]): List of data type strings (e.g., 'INT8', 'FP8', 'BF16', 'FP32')
                                             that this unit natively supports.
            peak_ops_per_datatype (dict[str, float]): A dictionary mapping datatype strings to their
                                                     peak operational capacity (ops/second, e.g., TOPS).
            energy_per_op_per_datatype (dict[str, float]): A dictionary mapping datatype strings to
                                                         the energy consumed per operation (Joules/op, e.g., pJ/op).
            memory_bandwidth_gbps (float): The simulated peak memory bandwidth in GB/s.
            latency_overhead_ns (float): A fixed latency overhead applied per operation,
                                         simulating pipeline stages, communication, etc. (in nanoseconds).
            num_cores (int): The number of parallel processing cores/units available.
            name (str): A human-readable name for the unit.
        """
        if not unit_id:
            raise ValueError("unit_id cannot be empty.")
        if not supported_datatypes:
            raise ValueError("supported_datatypes cannot be empty.")
        if not peak_ops_per_datatype or not all(dt in supported_datatypes for dt in peak_ops_per_datatype):
            raise ValueError("peak_ops_per_datatype must be provided for all supported_datatypes.")
        if not energy_per_op_per_datatype or not all(dt in supported_datatypes for dt in energy_per_op_per_datatype):
            raise ValueError("energy_per_op_per_datatype must be provided for all supported_datatypes.")
        if memory_bandwidth_gbps <= 0:
            raise ValueError("memory_bandwidth_gbps must be positive.")
        if latency_overhead_ns < 0:
            raise ValueError("latency_overhead_ns cannot be negative.")
        if num_cores <= 0:
            raise ValueError("num_cores must be positive.")

        self.unit_id = unit_id
        self.name = name
        self.supported_datatypes = set(supported_datatypes)
        self.peak_ops_per_datatype = peak_ops_per_datatype
        self.energy_per_op_per_datatype = energy_per_op_per_datatype
        self.memory_bandwidth = memory_bandwidth_gbps * 1e9 # Convert GB/s to B/s
        self.latency_overhead_ns = latency_overhead_ns
        self.num_cores = num_cores

        # Define byte sizes for common datatypes
        self._datatype_byte_size = {
            'INT8': 1, 'FP8': 1, 'BF16': 2, 'FP16': 2, 'FP32': 4, 'INT32': 4, 'BOOL': 1
        }

    def _get_datatype_byte_size(self, datatype: str) -> int:
        """Returns the byte size for a given datatype."""
        size = self._datatype_byte_size.get(datatype.upper())
        if size is None:
            raise ValueError(f"Unsupported datatype for byte size calculation: {datatype}")
        return size

    def supports_datatype(self, datatype: str) -> bool:
        """
        Checks if the hardware unit supports a given data type.

        Args:
            datatype (str): The data type string (e.g., 'INT8').

        Returns:
            bool: True if supported, False otherwise.
        """
        return datatype.upper() in self.supported_datatypes

    def _calculate_processing_latency_and_energy(self, num_ops: int, datatype: str) -> tuple[float, float]:
        """
        Calculates the computational latency and energy for a given number of operations.

        Args:
            num_ops (int): Total number of operations.
            datatype (str): The datatype of the operations.

        Returns:
            tuple[float, float]: A tuple containing (processing_latency_seconds, processing_energy_joules).
        """
        if not self.supports_datatype(datatype):
            raise ValueError(f"Datatype '{datatype}' not supported by this unit.")
        if num_ops < 0:
            raise ValueError("Number of operations cannot be negative.")
        if num_ops == 0:
            return 0.0, 0.0

        peak_ops_sec = self.peak_ops_per_datatype[datatype.upper()]
        energy_per_op = self.energy_per_op_per_datatype[datatype.upper()]

        # Compute-bound latency
        compute_latency = num_ops / peak_ops_sec if peak_ops_sec > 0 else float('inf')

        # Energy consumption
        energy_joules = num_ops * energy_per_op

        # Add per-operation overhead
        total_latency_overhead = num_ops * (self.latency_overhead_ns * 1e-9)

        return compute_latency + total_latency_overhead, energy_joules

    def simulate_memory_transfer(self, data_size_bytes: int) -> dict:
        """
        Simulates the time and energy cost of transferring data to/from the unit's memory.

        Args:
            data_size_bytes (int): The total size of data to be transferred in bytes.

        Returns:
            dict: A dictionary containing:
                  - 'latency_seconds': Estimated latency for memory transfer.
                  - 'energy_joules': Estimated energy for memory transfer (minimal for transfer itself).
                  - 'bandwidth_utilization': The achieved bandwidth utilization during transfer.
        """
        if data_size_bytes < 0:
            raise ValueError("Data size cannot be negative.")
        if data_size_bytes == 0:
            return {'latency_seconds': 0.0, 'energy_joules': 0.0, 'bandwidth_utilization': 0.0}

        transfer_latency = data_size_bytes / self.memory_bandwidth if self.memory_bandwidth > 0 else float('inf')
        # Simplified energy model for memory transfer: Assume a small fixed cost per byte
        transfer_energy = data_size_bytes * 1e-12 # e.g., 1 pJ/byte for memory access/transfer

        return {
            'latency_seconds': transfer_latency,
            'energy_joules': transfer_energy,
            'bandwidth_utilization': (data_size_bytes / transfer_latency) / self.memory_bandwidth if transfer_latency > 0 else 0.0
        }

    def simulate_matmul(self, M: int, N: int, K: int, datatype: str, sparsity: float = 0.0) -> dict:
        """
        Simulates a matrix multiplication operation (M x K) @ (K x N).

        Args:
            M (int): Number of rows in the first matrix.
            N (int): Number of columns in the second matrix.
            K (int): Number of columns in the first matrix / rows in the second matrix.
            datatype (str): The datatype for the computation (e.g., 'INT8', 'FP32').
            sparsity (float): The fraction of zero values (0.0 for dense, 0.9 for 90% sparse).
                              Applies to the operations count only, not data movement for now.

        Returns:
            dict: A dictionary containing simulated metrics:
                  - 'latency_seconds': Total estimated latency.
                  - 'energy_joules': Total estimated energy consumption.
                  - 'throughput_ops_per_sec': Achieved throughput.
                  - 'memory_access_latency_seconds': Latency due to data movement.
                  - 'compute_latency_seconds': Latency due to computation.
                  - 'total_ops': Total effective operations.
                  - 'datatype': The datatype used.
        """
        if not all(isinstance(dim, int) and dim > 0 for dim in [M, N, K]):
            raise ValueError("Matrix dimensions M, N, K must be positive integers.")
        if not (0.0 <= sparsity < 1.0): # Sparsity cannot be 1.0 because it would mean 0 operations
            raise ValueError("Sparsity must be between 0.0 and 1.0 (exclusive of 1.0).")
        if not self.supports_datatype(datatype):
            raise ValueError(f"Datatype '{datatype}' not supported by this unit.")

        # Total operations for a dense matrix multiplication (multiply-accumulates)
        total_dense_ops = 2 * M * N * K # Each element of output requires K multiplications and K-1 additions, approx 2K operations.
        effective_ops = total_dense_ops * (1.0 - sparsity)

        # Calculate compute latency and energy
        compute_latency, compute_energy = self._calculate_processing_latency_and_energy(
            int(effective_ops), datatype
        )

        # Calculate memory access
        byte_size = self._get_datatype_byte_size(datatype)
        input1_size = M * K * byte_size
        input2_size = K * N * byte_size
        output_size = M * N * byte_size
        total_data_bytes = input1_size + input2_size + output_size # Simplified: inputs loaded once, output written once

        mem_transfer_results = self.simulate_memory_transfer(total_data_bytes)
        mem_latency = mem_transfer_results['latency_seconds']
        mem_energy = mem_transfer_results['energy_joules']

        # Total latency is max of compute and memory access (simplified model for pipelining)
        # For a more complex model, this would involve detailed scheduling.
        total_latency = max(compute_latency, mem_latency)
        total_energy = compute_energy + mem_energy
        throughput = effective_ops / total_latency if total_latency > 0 else float('inf')

        return {
            'latency_seconds': total_latency,
            'energy_joules': total_energy,
            'throughput_ops_per_sec': throughput,
            'memory_access_latency_seconds': mem_latency,
            'compute_latency_seconds': compute_latency,
            'total_ops': effective_ops,
            'datatype': datatype,
            'sparsity': sparsity,
            'unit_id': self.unit_id
        }

    def simulate_elementwise_op(self, num_elements: int, datatype: str) -> dict:
        """
        Simulates an element-wise operation (e.g., ReLU, ADD, MUL).

        Args:
            num_elements (int): The total number of elements involved in the operation.
            datatype (str): The datatype for the computation.

        Returns:
            dict: A dictionary containing simulated metrics:
                  - 'latency_seconds': Total estimated latency.
                  - 'energy_joules': Total estimated energy consumption.
                  - 'throughput_ops_per_sec': Achieved throughput.
                  - 'memory_access_latency_seconds': Latency due to data movement.
                  - 'compute_latency_seconds': Latency due to computation.
                  - 'total_ops': Total effective operations.
                  - 'datatype': The datatype used.
        """
        if not isinstance(num_elements, int) or num_elements <= 0:
            raise ValueError("Number of elements must be a positive integer.")
        if not self.supports_datatype(datatype):
            raise ValueError(f"Datatype '{datatype}' not supported by this unit.")

        # Each element contributes one operation (e.g., one addition, one ReLU)
        effective_ops = num_elements

        # Calculate compute latency and energy
        compute_latency, compute_energy = self._calculate_processing_latency_and_energy(
            effective_ops, datatype
        )

        # Calculate memory access (read input, write output)
        byte_size = self._get_datatype_byte_size(datatype)
        total_data_bytes = num_elements * byte_size * 2 # Read input, write output

        mem_transfer_results = self.simulate_memory_transfer(total_data_bytes)
        mem_latency = mem_transfer_results['latency_seconds']
        mem_energy = mem_transfer_results['energy_joules']

        total_latency = max(compute_latency, mem_latency)
        total_energy = compute_energy + mem_energy
        throughput = effective_ops / total_latency if total_latency > 0 else float('inf')

        return {
            'latency_seconds': total_latency,
            'energy_joules': total_energy,
            'throughput_ops_per_sec': throughput,
            'memory_access_latency_seconds': mem_latency,
            'compute_latency_seconds': compute_latency,
            'total_ops': effective_ops,
            'datatype': datatype,
            'unit_id': self.unit_id
        }

    def get_info(self) -> dict:
        """Returns a dictionary with key information about the hardware unit."""
        return {
            'unit_id': self.unit_id,
            'name': self.name,
            'supported_datatypes': sorted(list(self.supported_datatypes)),
            'peak_ops_per_datatype': self.peak_ops_per_datatype,
            'energy_per_op_per_datatype': self.energy_per_op_per_datatype,
            'memory_bandwidth_gbps': self.memory_bandwidth / 1e9,
            'latency_overhead_ns': self.latency_overhead_ns,
            'num_cores': self.num_cores
        }