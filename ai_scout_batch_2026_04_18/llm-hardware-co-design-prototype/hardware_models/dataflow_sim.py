class DataflowAccelerator:
    """
    Simulates a specialized AI accelerator with a dataflow architecture,
    emphasizing pipelined execution and explicit data movement.
    """

    def __init__(self,
                 name="GenericDataflowAccelerator",
                 num_compute_units=128,
                 compute_unit_frequency_ghz=1.0, # GHz
                 on_chip_memory_bandwidth_gbps=1024, # GB/s
                 off_chip_memory_bandwidth_gbps=100, # GB/s
                 scratchpad_bandwidth_gbps=2048, # GB/s
                 op_energy_per_flop_pj={'FP32': 20, 'FP16': 5, 'FP8': 2, 'INT8': 1}, # pJ/flop
                 mem_energy_per_byte_pj={'on-chip': 0.1, 'off-chip': 1.0, 'scratchpad': 0.05}, # pJ/byte
                 sparse_compute_efficiency=0.8, # Efficiency factor for sparse ops (e.g., 0.8 means 20% savings)
                 quant_overhead_factor=0.9 # Factor for ops with quantization support (e.g., 0.9 means 10% saving)
                ):
        self.name = name
        self.num_compute_units = num_compute_units
        self.compute_unit_frequency_ghz = compute_unit_frequency_ghz
        self.on_chip_memory_bandwidth_gbps = on_chip_memory_bandwidth_gbps
        self.off_chip_memory_bandwidth_gbps = off_chip_memory_bandwidth_gbps
        self.scratchpad_bandwidth_gbps = scratchpad_bandwidth_gbps
        self.op_energy_per_flop_pj = op_energy_per_flop_pj
        self.mem_energy_per_byte_pj = mem_energy_per_byte_pj
        self.sparse_compute_efficiency = sparse_compute_efficiency
        self.quant_overhead_factor = quant_overhead_factor

        # Derived properties
        self.compute_unit_flops_per_cycle = 8 # Assuming 8 FLOPs/cycle per CU (e.g., SIMD lanes)
        self.total_max_gflops = self.num_compute_units * self.compute_unit_flops_per_cycle * self.compute_unit_frequency_ghz
        self.cycle_time_ns = 1 / self.compute_unit_frequency_ghz # ns

        self._supported_data_types = ['FP32', 'FP16', 'FP8', 'INT8']
        self._supported_features = {'quantization': True, 'sparse_ops': True, 'dynamic_batching': True}

        self._op_cost_model = {
            'GEMM': lambda M, N, K: 2 * M * N * K, # FLOPs for matrix multiplication
            'ADD': lambda *shape: max(1, *shape),  # Element-wise add
            'MUL': lambda *shape: max(1, *shape),  # Element-wise multiply
            'ACTIVATION': lambda *shape: max(1, *shape) # Element-wise activation
        }

    def get_supported_data_types(self):
        """Returns a list of data types supported by the accelerator."""
        return list(self._supported_data_types)

    def get_supported_features(self):
        """Returns a dictionary of features supported by the accelerator."""
        return dict(self._supported_features)

    def _get_data_type_size(self, data_type):
        """Returns size in bytes for a given data type."""
        if data_type == 'FP32': return 4
        if data_type == 'FP16': return 2
        if data_type == 'FP8': return 1
        if data_type == 'INT8': return 1
        raise ValueError(f"Unsupported data type: {data_type}")

    def simulate_operation(self, op_type: str, operands_shape: tuple, data_type: str, sparsity: float = 0.0):
        """
        Simulates the computational cost (latency, energy) for a single operation.

        Args:
            op_type (str): Type of operation (e.g., 'GEMM', 'ADD', 'MUL', 'ACTIVATION').
            operands_shape (tuple): Shape of the operands (e.g., (M, N, K) for GEMM, (N,) for vector).
            data_type (str): Data type of the operation (e.g., 'FP32', 'INT8').
            sparsity (float): Sparsity level (0.0 to 1.0). Only impacts ops like GEMM if sparse_ops is True.

        Returns:
            dict: Simulated metrics {'latency_ns': float, 'energy_pj': float, 'flops': int}.
        Raises:
            ValueError: If op_type or data_type is not supported.
        """
        if op_type not in self._op_cost_model:
            raise ValueError(f"Unsupported operation type: {op_type}")
        if data_type not in self._supported_data_types:
            raise ValueError(f"Unsupported data type for operation: {data_type}")
        if not all(isinstance(d, int) and d > 0 for d in operands_shape):
            raise ValueError(f"Invalid operands_shape: {operands_shape}. All dimensions must be positive integers.")

        flops = self._op_cost_model[op_type](*operands_shape)

        # Apply sparsity if applicable and feature is supported
        if self._supported_features['sparse_ops'] and sparsity > 0.0 and op_type == 'GEMM':
            flops_effective = flops * (1.0 - sparsity * self.sparse_compute_efficiency)
        else:
            flops_effective = flops

        # Calculate theoretical minimum cycles based on peak FLOPs
        cycles_needed = flops_effective / self.total_max_gflops
        latency_ns = cycles_needed * 1000 # Convert cycles to ns

        # Energy calculation
        energy_per_flop = self.op_energy_per_flop_pj.get(data_type)
        if energy_per_flop is None:
            # Fallback for unsupported data types, or assume FP32 cost
            energy_per_flop = self.op_energy_per_flop_pj['FP32']
            print(f"Warning: No specific energy cost for {data_type}. Using FP32 energy cost.")

        energy_pj = flops_effective * energy_per_flop

        # Apply quantization overhead/benefit factor
        if data_type in ['FP8', 'INT8'] and self._supported_features['quantization']:
            latency_ns *= self.quant_overhead_factor # Assume some minor reduction due to specialized units
            energy_pj *= self.quant_overhead_factor

        return {
            'latency_ns': latency_ns,
            'energy_pj': energy_pj,
            'flops': flops_effective
        }

    def simulate_memory_access(self, size_bytes: int, access_type: str = "read", locality: str = "on-chip"):
        """
        Simulates the cost (latency, energy) for memory access.

        Args:
            size_bytes (int): Amount of data to access in bytes.
            access_type (str): Type of access ('read', 'write').
            locality (str): Where the memory resides ('on-chip', 'off-chip', 'scratchpad').

        Returns:
            dict: Simulated metrics {'latency_ns': float, 'energy_pj': float}.
        Raises:
            ValueError: If size_bytes is non-positive, or locality/access_type is not supported.
        """
        if size_bytes <= 0:
            raise ValueError(f"Memory access size_bytes must be positive, got {size_bytes}")
        if locality not in ['on-chip', 'off-chip', 'scratchpad']:
            raise ValueError(f"Unsupported memory locality: {locality}")
        if access_type not in ['read', 'write']:
            raise ValueError(f"Unsupported memory access type: {access_type}")

        # Bandwidth is in GB/s, convert to B/ns (GB/s * 10^9 B/GB * 10^-9 s/ns = B/ns)
        bandwidth_gbps = {
            'on-chip': self.on_chip_memory_bandwidth_gbps,
            'off-chip': self.off_chip_memory_bandwidth_gbps,
            'scratchpad': self.scratchpad_bandwidth_gbps
        }.get(locality)

        if bandwidth_gbps is None:
            raise ValueError(f"Unknown bandwidth for locality: {locality}")

        bandwidth_bps = bandwidth_gbps * 1e9 # B/s
        bandwidth_b_per_ns = bandwidth_bps / 1e9 # B/ns

        if bandwidth_b_per_ns == 0:
            # Avoid division by zero if bandwidth is somehow set to 0
            latency_ns = float('inf')
        else:
            latency_ns = size_bytes / bandwidth_b_per_ns

        # Energy per byte
        energy_per_byte = self.mem_energy_per_byte_pj.get(locality)
        if energy_per_byte is None:
            # Fallback for unknown locality
            energy_per_byte = self.mem_energy_per_byte_pj['off-chip']
            print(f"Warning: No specific energy cost for {locality}. Using off-chip memory energy cost.")

        energy_pj = size_bytes * energy_per_byte

        # Read/write might have slightly different energy costs, but for this prototype, we'll keep it simple.
        # Could add a factor here if needed, e.g., if access_type == 'write': energy_pj *= 1.1

        return {
            'latency_ns': latency_ns,
            'energy_pj': energy_pj
        }

    def estimate_throughput(self, op_type: str, operands_shape: tuple, data_type: str, sparsity: float = 0.0):
        """
        Estimates the throughput (operations per second) for a given operation.

        Args:
            op_type (str): Type of operation.
            operands_shape (tuple): Shape of the operands.
            data_type (str): Data type of the operation.
            sparsity (float): Sparsity level.

        Returns:
            float: Estimated throughput in ops/second.
            float: Estimated throughput in FLOPs/second.
        Raises:
            ValueError: If simulation fails.
        """
        try:
            result = self.simulate_operation(op_type, operands_shape, data_type, sparsity)
            latency_ns = result['latency_ns']
            flops = result['flops']

            if latency_ns == 0:
                # Avoid division by zero for extremely fast operations, assume max possible throughput
                ops_per_sec = float('inf')
                flops_per_sec = float('inf')
            else:
                latency_s = latency_ns / 1e9 # Convert ns to seconds
                ops_per_sec = 1.0 / latency_s
                flops_per_sec = flops / latency_s
            return ops_per_sec, flops_per_sec
        except Exception as e:
            raise ValueError(f"Failed to estimate throughput: {e}") from e

    def __str__(self):
        return (f"{self.name} (Dataflow Accelerator)\n"
                f"  - Max GFLOPs: {self.total_max_gflops:.2f}\n"
                f"  - On-chip BW: {self.on_chip_memory_bandwidth_gbps} GB/s\n"
                f"  - Off-chip BW: {self.off_chip_memory_bandwidth_gbps} GB/s\n"
                f"  - Supported Data Types: {', '.join(self._supported_data_types)}")