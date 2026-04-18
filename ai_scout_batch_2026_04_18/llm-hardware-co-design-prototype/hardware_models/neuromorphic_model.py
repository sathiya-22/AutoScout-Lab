```python
import math

class NeuromorphicModel:
    """
    Simulates a neuromorphic hardware accelerator tailored for LLM inference patterns.
    This model abstracts key aspects like computational costs (latency, energy),
    memory access patterns, and support for features like sparsity and quantization.
    """

    def __init__(self,
                 num_cores: int = 256,
                 neurons_per_core: int = 1024,
                 synapses_per_neuron: int = 256,
                 base_energy_per_spike_pJ: float = 0.1,  # pJ per spike
                 base_neuron_op_latency_ns: float = 10.0, # ns per neuron activation
                 on_chip_memory_bandwidth_GBps: float = 500.0,
                 off_chip_memory_bandwidth_GBps: float = 100.0,
                 max_supported_batch_size: int = 64,
                 supports_dynamic_sparsity: bool = True,
                 supported_precisions: list = None):
        """
        Initializes the NeuromorphicModel with key architectural parameters.

        Args:
            num_cores (int): Number of processing cores/neuro-clusters.
            neurons_per_core (int): Number of neurons per core.
            synapses_per_neuron (int): Average number of synapses connected to a neuron.
            base_energy_per_spike_pJ (float): Base energy consumed for a single spike event in picojoules.
            base_neuron_op_latency_ns (float): Base latency for a neuron to process an input and potentially spike.
            on_chip_memory_bandwidth_GBps (float): Simulated on-chip memory bandwidth in GB/s.
            off_chip_memory_bandwidth_GBps (float): Simulated off-chip memory bandwidth in GB/s.
            max_supported_batch_size (int): Maximum batch size the hardware can efficiently handle.
            supports_dynamic_sparsity (bool): Whether the hardware natively benefits from dynamic sparsity.
            supported_precisions (list): List of supported data precisions (e.g., 'INT8', 'FP16', 'BIN').
        """
        if supported_precisions is None:
            supported_precisions = ['INT8', 'BIN', 'FP16'] # Binary for neuromorphic spike events, INT8/FP16 for inputs/outputs
        
        self.num_cores = num_cores
        self.neurons_per_core = neurons_per_core
        self.total_neurons = num_cores * neurons_per_core
        self.synapses_per_neuron = synapses_per_neuron
        self.base_energy_per_spike_pJ = base_energy_per_spike_pJ
        self.base_neuron_op_latency_ns = base_neuron_op_latency_ns
        self.on_chip_memory_bandwidth_GBps = on_chip_memory_bandwidth_GBps
        self.off_chip_memory_bandwidth_GBps = off_chip_memory_bandwidth_GBps
        self.max_supported_batch_size = max_supported_batch_size
        self.supports_dynamic_sparsity = supports_dynamic_sparsity
        self.supported_precisions = [p.upper() for p in supported_precisions]

        # Cost multipliers (simplified for prototyping)
        self.precision_energy_multiplier = {
            'BIN': 1.0,  # Binary spikes are most native
            'INT8': 1.5,
            'FP16': 3.0,
            'FP32': 8.0   # High cost for non-native precision
        }
        self.precision_latency_multiplier = {
            'BIN': 1.0,
            'INT8': 1.2,
            'FP16': 2.0,
            'FP32': 4.0
        }
        # Sparsity advantage: how much latency/energy reduces with sparsity (0.0=no advantage, 1.0=full reduction)
        self.sparsity_advantage_factor_latency = 0.8
        self.sparsity_advantage_factor_energy = 0.9

    def _get_precision_multipliers(self, precision: str) -> tuple[float, float]:
        """Returns energy and latency multipliers for a given precision."""
        upper_precision = precision.upper()
        if upper_precision not in self.supported_precisions:
            # Fallback to nearest supported or highest cost if unsupported
            print(f"Warning: Precision {precision} not natively supported. Using FP32 cost model.")
            return self.precision_energy_multiplier.get('FP32', 10.0), self.precision_latency_multiplier.get('FP32', 5.0)
        return (self.precision_energy_multiplier.get(upper_precision, 1.0),
                self.precision_latency_multiplier.get(upper_precision, 1.0))

    def _simulate_latency(self, operation_type: str, input_elements: int,
                          sparsity: float = 0.0, precision: str = 'INT8',
                          batch_size: int = 1) -> float:
        """
        Simulates the latency for a given operation on the neuromorphic model in nanoseconds.
        This is a simplified cost model.

        Args:
            operation_type (str): Type of operation (e.g., 'matrix_multiply', 'add', 'activation', 'memory_transfer').
            input_elements (int): Total number of elements involved in the primary input of the operation.
            sparsity (float): Input sparsity (0.0 to 1.0, 0.0 means dense).
            precision (str): Data precision ('INT8', 'FP16', 'BIN').
            batch_size (int): Current batch size.
        Returns:
            float: Simulated latency in nanoseconds.
        """
        if not (0.0 <= sparsity <= 1.0):
            raise ValueError("Sparsity must be between 0.0 and 1.0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        precision_energy_mult, precision_latency_mult = self._get_precision_multipliers(precision)

        base_latency = 0.0

        # Simulate batching overhead if exceeding max supported
        batch_scale_factor = math.ceil(batch_size / self.max_supported_batch_size) if batch_size > self.max_supported_batch_size else 1.0

        effective_sparsity = sparsity if self.supports_dynamic_sparsity else 0.0
        # For neuromorphic, higher sparsity should lead to lower latency
        sparsity_reduction_factor = 1.0 - (effective_sparsity * self.sparsity_advantage_factor_latency)

        if operation_type == 'matrix_multiply':
            # Simplified: Latency scales with input_elements (roughly N*K*M for N*K @ K*M)
            # and is influenced by neuron/synapse capacity
            ops_per_neuron_time_step = self.synapses_per_neuron # rough estimate
            # Consider total ops as product of input elements and a factor related to output elements
            # A common MM (M,K)x(K,N) involves M*K*N MACs
            # For simplicity, let's assume input_elements is roughly M*K or K*N.
            # We scale it by a factor for the actual computation complexity.
            complexity_factor = math.log(input_elements + 1) * 0.1 # logarithmic scaling for complexity

            if input_elements == 0: # Avoid log(0)
                 base_latency = self.base_neuron_op_latency_ns * self.total_neurons # Minimum latency for an empty op
            else:
                 # Assume each neuron can handle some number of "effective" synapses per time step
                 # Total "work" is proportional to input_elements.
                 # Division by total_neurons gives work per neuron, then multiply by latency per neuron op
                 base_latency = (input_elements / self.total_neurons) * self.base_neuron_op_latency_ns * complexity_factor
                 base_latency = max(base_latency, self.base_neuron_op_latency_ns) # Ensure minimum latency

        elif operation_type == 'add' or operation_type == 'element_wise':
            # Element-wise operations, scales with input size but less aggressively
            base_latency = (input_elements / self.total_neurons) * self.base_neuron_op_latency_ns * 0.1
            base_latency = max(base_latency, self.base_neuron_op_latency_ns * 0.1) # Minimum

        elif operation_type == 'activation':
            # Activation functions (ReLU, GeLU, Softmax). Each neuron can perform activation.
            # Softmax might involve global communication, adding latency.
            activation_factor = 0.5 if input_elements <= self.total_neurons else (input_elements / self.total_neurons)
            base_latency = activation_factor * self.base_neuron_op_latency_ns
            base_latency = max(base_latency, self.base_neuron_op_latency_ns * 0.5) # Minimum

        elif operation_type == 'memory_transfer':
            # This is handled by _simulate_memory_access; a direct op_type for it here is redundant.
            # Return 0 or raise error if it's not meant to be called this way.
            return self._simulate_memory_access(input_elements * self._get_bytes_per_element(precision), 'read', batch_size)

        else:
            # Generic operation cost, default to a base cost
            base_latency = (input_elements / self.total_neurons) * self.base_neuron_op_latency_ns * 0.5
            base_latency = max(base_latency, self.base_neuron_op_latency_ns * 0.2)

        # Apply precision, sparsity, and batching factors
        final_latency = base_latency * precision_latency_mult * sparsity_reduction_factor * batch_scale_factor
        return max(final_latency, self.base_neuron_op_latency_ns * 0.01) # Ensure some minimal latency

    def _simulate_energy(self, operation_type: str, input_elements: int,
                         sparsity: float = 0.0, precision: str = 'INT8',
                         batch_size: int = 1) -> float:
        """
        Simulates the energy consumption for a given operation in picojoules.
        This is a simplified cost model.

        Args:
            operation_type (str): Type of operation.
            input_elements (int): Total number of elements involved.
            sparsity (float): Input sparsity (0.0 to 1.0).
            precision (str): Data precision ('INT8', 'FP16', 'BIN').
            batch_size (int): Current batch size.
        Returns:
            float: Simulated energy in picojoules.
        """
        if not (0.0 <= sparsity <= 1.0):
            raise ValueError("Sparsity must be between 0.0 and 1.0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        precision_energy_mult, _ = self._get_precision_multipliers(precision)

        base_energy = 0.0
        effective_sparsity = sparsity if self.supports_dynamic_sparsity else 0.0
        # For neuromorphic, higher sparsity should lead to lower energy
        sparsity_reduction_factor = 1.0 - (effective_sparsity * self.sparsity_advantage_factor_energy)

        # Assume energy is related to "active" operations or spikes
        # A rough estimate for total spikes/active operations
        total_active_events = input_elements * batch_size

        if operation_type == 'matrix_multiply':
            # Each "effective" MAC operation might correspond to several spike events.
            # Scales with input_elements and a complexity factor.
            complexity_factor = math.log(input_elements + 1) * 0.05
            base_energy = total_active_events * self.base_energy_per_spike_pJ * self.synapses_per_neuron * complexity_factor
        elif operation_type == 'add' or operation_type == 'element_wise':
            # Simpler ops, fewer spikes per element
            base_energy = total_active_events * self.base_energy_per_spike_pJ * 0.5
        elif operation_type == 'activation':
            # Each neuron potentially spikes
            base_energy = total_active_events * self.base_energy_per_spike_pJ
        elif operation_type == 'memory_transfer':
            # This is handled by _simulate_memory_access
            return self._simulate_memory_access(input_elements * self._get_bytes_per_element(precision), 'read', batch_size, return_energy=True)
        else:
            base_energy = total_active_events * self.base_energy_per_spike_pJ * 0.8

        final_energy = base_energy * precision_energy_mult * sparsity_reduction_factor
        return max(final_energy, self.base_energy_per_spike_pJ * batch_size) # Ensure minimal energy

    def _get_bytes_per_element(self, precision: str) -> int:
        """Helper to get bytes per element for a given precision."""
        upper_precision = precision.upper()
        if upper_precision == 'BIN': return 1/8 # 1 bit
        if upper_precision == 'INT8': return 1
        if upper_precision == 'FP8': return 1 # Assuming custom FP8 support
        if upper_precision == 'FP16': return 2
        if upper_precision == 'INT16': return 2
        if upper_precision == 'FP32': return 4
        return 4 # Default to FP32

    def _simulate_memory_access(self, data_size_bytes: int, access_type: str = 'read',
                                batch_size: int = 1, return_energy: bool = False) -> float:
        """
        Simulates memory access cost (latency or energy).
        Assumes data often needs to be transferred to/from cores.
        Prioritizes on-chip memory for smaller transfers.

        Args:
            data_size_bytes (int): Size of data to be accessed in bytes.
            access_type (str): 'read' or 'write'.
            batch_size (int): Current batch size.
            return_energy (bool): If True, returns energy in pJ; otherwise, latency in ns.
        Returns:
            float: Simulated latency in nanoseconds or energy in picojoules.
        """
        if data_size_bytes < 0:
            raise ValueError("Data size must be non-negative.")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        # Heuristic: Small transfers stay on-chip, large ones go off-chip.
        # This is a simplification; a real model would track memory hierarchy.
        on_chip_threshold_bytes = self.total_neurons * self.synapses_per_neuron * 2 # Roughly total synapse memory * 2 bytes/synapse
        
        # Scale for batching, as memory access might be pipelined or parallelized per batch item
        effective_data_size = data_size_bytes # Assume data_size_bytes already includes batching when passed for an operation

        if effective_data_size <= on_chip_threshold_bytes:
            bandwidth_GBps = self.on_chip_memory_bandwidth_GBps
            # On-chip memory access is often energy-efficient
            energy_per_byte_pJ = 0.5 # pJ/byte for on-chip
        else:
            bandwidth_GBps = self.off_chip_memory_bandwidth_GBps
            # Off-chip memory access is more energy intensive
            energy_per_byte_pJ = 5.0 # pJ/byte for off-chip

        # Convert bandwidth to bytes/ns for latency calculation
        bandwidth_bytes_per_ns = bandwidth_GBps * 1e9 / 1e9 # GB/s * (1e9 bytes/GB) / (1e9 ns/s) = GBps
        if bandwidth_bytes_per_ns == 0: bandwidth_bytes_per_ns = 1e-9 # Avoid division by zero

        latency_ns = (effective_data_size / bandwidth_bytes_per_ns) if effective_data_size > 0 else 0.0
        # Add a small base access latency (e.g., for address decoding)
        latency_ns += 5.0

        energy_pJ = effective_data_size * energy_per_byte_pJ
        # A minimal energy cost even for zero data to reflect control logic
        energy_pJ = max(energy_pJ, energy_per_byte_pJ * 0.1)

        if return_energy:
            return energy_pJ
        return latency_ns

    def estimate_operation_cost(self, operation_type: str, input_shape: tuple,
                                sparsity: float = 0.0, precision: str = 'INT8',
                                batch_size: int = 1) -> dict:
        """
        Estimates the computational cost for a given LLM operation.

        Args:
            operation_type (str): The type of LLM operation (e.g., 'matrix_multiply', 'add', 'activation').
            input_shape (tuple): Shape of the primary input tensor (e.g., (batch, seq_len, hidden_dim)).
                                 For memory_transfer, this should be (data_size_bytes,).
            sparsity (float): Estimated sparsity of the input activations or weights (0.0 to 1.0).
            precision (str): Data precision ('INT8', 'FP16', 'BIN').
            batch_size (int): The current batch size for the operation.

        Returns:
            dict: A dictionary containing 'latency_ns', 'energy_pJ', and 'throughput_ops_per_sec'.
        """
        if not input_shape:
            raise ValueError("Input shape cannot be empty.")
        if not isinstance(input_shape, tuple):
            raise TypeError("Input shape must be a tuple.")
        if any(d <= 0 for d in input_shape):
            raise ValueError("Dimensions in input_shape must be positive.")
        if operation_type not in ['matrix_multiply', 'add', 'element_wise', 'activation', 'memory_transfer']:
            print(f"Warning: Unknown operation type '{operation_type}'. Using generic cost model.")

        if operation_type == 'memory_transfer':
            # For memory_transfer, input_shape is assumed to be (data_size_bytes,)
            if len(input_shape) != 1:
                raise ValueError("For 'memory_transfer', input_shape must be (data_size_bytes,).")
            data_size_bytes = input_shape[0]
            latency_ns = self._simulate_memory_access(data_size_bytes, 'read', batch_size)
            energy_pJ = self._simulate_memory_access(data_size_bytes, 'read', batch_size, return_energy=True)
            # Throughput for memory transfer is usually MB/s or GB/s, not ops/sec.
            # For consistency, we can define a "transfer op" as transferring 'data_size_bytes'
            # So, 1 / (latency_s) * batch_size
            throughput_ops_per_sec = (batch_size / (latency_ns / 1e9)) if latency_ns > 0 else float('inf')
        else:
            # Calculate total input elements (e.g., for a tensor)
            input_elements = math.prod(input_shape)

            # Separate compute and memory costs
            compute_latency_ns = self._simulate_latency(operation_type, input_elements, sparsity, precision, batch_size)
            compute_energy_pJ = self._simulate_energy(operation_type, input_elements, sparsity, precision, batch_size)

            # Estimate data movement cost associated with the operation
            # Assuming inputs and outputs need to be moved for this operation
            # Simplified: data size is input_elements * bytes_per_element
            bytes_per_element = self._get_bytes_per_element(precision)
            data_movement_bytes = input_elements * bytes_per_element * 2 # Input + (approx) Output
            
            memory_latency_ns = self._simulate_memory_access(data_movement_bytes, 'read_write', batch_size)
            memory_energy_pJ = self._simulate_memory_access(data_movement_bytes, 'read_write', batch_size, return_energy=True)

            latency_ns = compute_latency_ns + memory_latency_ns
            energy_pJ = compute_energy_pJ + memory_energy_pJ

            # Throughput is usually operations per second.
            # A single LLM operation on input_shape is considered 1 "op" for throughput calculation.
            throughput_ops_per_sec = (batch_size / (latency_ns / 1e9)) if latency_ns > 0 else float('inf')

        return {
            'latency_ns': latency_ns,
            'energy_pJ': energy_pJ,
            'throughput_ops_per_sec': throughput_ops_per_sec
        }

    def get_supported_precisions(self) -> list[str]:
        """Returns a list of data precisions supported by the hardware."""
        return self.supported_precisions

    def supports_sparse_computation(self) -> bool:
        """Indicates if the hardware can natively leverage sparse computations."""
        return self.supports_dynamic_sparsity

    def get_memory_bandwidth(self, is_on_chip: bool = False) -> float:
        """
        Returns the simulated memory bandwidth.

        Args:
            is_on_chip (bool): If True, returns on-chip bandwidth; otherwise, off-chip.
        Returns:
            float: Bandwidth in GB/s.
        """
        return self.on_chip_memory_bandwidth_GBps if is_on_chip else self.off_chip_memory_bandwidth_GBps

    def get_max_batch_size(self) -> int:
        """Returns the maximum batch size the hardware efficiently supports."""
        return self.max_supported_batch_size

if __name__ == '__main__':
    # Example Usage:
    neuromorphic_chip = NeuromorphicModel(
        num_cores=512,
        neurons_per_core=2048,
        synapses_per_neuron=512,
        base_energy_per_spike_pJ=0.05,
        base_neuron_op_latency_ns=5.0,
        on_chip_memory_bandwidth_GBps=1000.0,
        off_chip_memory_bandwidth_GBps=200.0,
        max_supported_batch_size=128,
        supported_precisions=['BIN', 'INT8', 'FP8']
    )

    print(f"Neuromorphic Model Initialized:")
    print(f"  Supports sparse computation: {neuromorphic_chip.supports_sparse_computation()}")
    print(f"  Supported precisions: {neuromorphic_chip.get_supported_precisions()}")
    print(f"  Max efficient batch size: {neuromorphic_chip.get_max_batch_size()}")
    print(f"  On-chip memory bandwidth: {neuromorphic_chip.get_memory_bandwidth(True)} GB/s")
    print(f"  Off-chip memory bandwidth: {neuromorphic_chip.get_memory_bandwidth(False)} GB/s\n")

    # Simulate an Attention Block's Matrix Multiply (e.g., QK^T or SoftmaxV)
    # Common dimensions: (batch_size, num_heads, seq_len, head_dim)
    batch = 16
    seq_len = 2048
    hidden_dim = 1024
    head_dim = 64
    num_heads = hidden_dim // head_dim
    mm_input_shape_qk = (batch, num_heads, seq_len, head_dim) # Example for QK^T, input elements roughly seq_len * head_dim * seq_len
    mm_input_elements = batch * num_heads * seq_len * head_dim # A simplified representation of input elements count

    print(f"Simulating Matrix Multiply (Attention block) with input elements: {mm_input_elements}")
    print(f"  Input Shape: {mm_input_shape_qk}")

    # Dense FP16 calculation
    cost_dense_fp16 = neuromorphic_chip.estimate_operation_cost(
        operation_type='matrix_multiply',
        input_shape=mm_input_shape_qk, # Using the full shape to represent the complexity
        sparsity=0.0,
        precision='FP16',
        batch_size=batch
    )
    print(f"  Dense FP16: Latency={cost_dense_fp16['latency_ns']:.2f} ns, Energy={cost_dense_fp16['energy_pJ']:.2f} pJ, Throughput={cost_dense_fp16['throughput_ops_per_sec']:.2f} ops/sec")

    # Sparse INT8 calculation (e.g., post-activation sparsity)
    cost_sparse_int8 = neuromorphic_chip.estimate_operation_cost(
        operation_type='matrix_multiply',
        input_shape=mm_input_shape_qk,
        sparsity=0.7, # 70% sparse
        precision='INT8',
        batch_size=batch
    )
    print(f"  Sparse INT8 (70%): Latency={cost_sparse_int8['latency_ns']:.2f} ns, Energy={cost_sparse_int8['energy_pJ']:.2f} pJ, Throughput={cost_sparse_int8['throughput_ops_per_sec']:.2f} ops/sec")

    # Binary precision (native to neuromorphic, representing spike events)
    cost_bin_sparse = neuromorphic_chip.estimate_operation_cost(
        operation_type='matrix_multiply',
        input_shape=mm_input_shape_qk,
        sparsity=0.9, # Very sparse, highly efficient
        precision='BIN',
        batch_size=batch
    )
    print(f"  Sparse BIN (90%): Latency={cost_bin_sparse['latency_ns']:.2f} ns, Energy={cost_bin_sparse['energy_pJ']:.2f} pJ, Throughput={cost_bin_sparse['throughput_ops_per_sec']:.2f} ops/sec\n")


    # Simulate a Feed-Forward Layer Activation (e.g., GeLU)
    ffn_input_shape = (batch, seq_len, hidden_dim * 4) # Expanded dimension
    ffn_input_elements = math.prod(ffn_input_shape)

    print(f"Simulating Activation (GeLU) with input elements: {ffn_input_elements}")
    cost_activation_dense = neuromorphic_chip.estimate_operation_cost(
        operation_type='activation',
        input_shape=ffn_input_shape,
        sparsity=0.0,
        precision='FP8',
        batch_size=batch
    )
    print(f"  Dense FP8: Latency={cost_activation_dense['latency_ns']:.2f} ns, Energy={cost_activation_dense['energy_pJ']:.2f} pJ, Throughput={cost_activation_dense['throughput_ops_per_sec']:.2f} ops/sec")

    cost_activation_sparse = neuromorphic_chip.estimate_operation_cost(
        operation_type='activation',
        input_shape=ffn_input_shape,
        sparsity=0.8,
        precision='INT8',
        batch_size=batch
    )
    print(f"  Sparse INT8 (80%): Latency={cost_activation_sparse['latency_ns']:.2f} ns, Energy={cost_activation_sparse['energy_pJ']:.2f} pJ, Throughput={cost_activation_sparse['throughput_ops_per_sec']:.2f} ops/sec\n")

    # Simulate Memory Transfer (e.g., loading model weights)
    model_weights_size_bytes = 100 * 1024 * 1024 # 100 MB
    print(f"Simulating Memory Transfer of {model_weights_size_bytes / (1024*1024):.2f} MB")
    cost_mem_transfer = neuromorphic_chip.estimate_operation_cost(
        operation_type='memory_transfer',
        input_shape=(model_weights_size_bytes,),
        batch_size=1
    )
    print(f"  Memory Transfer: Latency={cost_mem_transfer['latency_ns']:.2f} ns, Energy={cost_mem_transfer['energy_pJ']:.2f} pJ, Throughput={cost_mem_transfer['throughput_ops_per_sec']:.2f} 'transfer ops'/sec\n")

    # Error handling test cases
    try:
        neuromorphic_chip.estimate_operation_cost('matrix_multiply', (16, 0, 1024), sparsity=0.5)
    except ValueError as e:
        print(f"Error handling test (input_shape with zero dim): {e}")

    try:
        neuromorphic_chip.estimate_operation_cost('matrix_multiply', (16, 256, 1024), sparsity=1.5)
    except ValueError as e:
        print(f"Error handling test (invalid sparsity): {e}")

    try:
        neuromorphic_chip.estimate_operation_cost('matrix_multiply', (16, 256, 1024), batch_size=0)
    except ValueError as e:
        print(f"Error handling test (invalid batch_size): {e}")

    try:
        neuromorphic_chip.estimate_operation_cost('memory_transfer', (1024, 2), batch_size=1)
    except ValueError as e:
        print(f"Error handling test (invalid memory_transfer shape): {e}")

    # Test unsupported precision
    cost_unsupported_precision = neuromorphic_chip.estimate_operation_cost(
        operation_type='add',
        input_shape=(batch, 1024),
        precision='FP32',
        batch_size=batch
    )
    print(f"\nUnsupported Precision (FP32) Add: Latency={cost_unsupported_precision['latency_ns']:.2f} ns, Energy={cost_unsupported_precision['energy_pJ']:.2f} pJ")
```