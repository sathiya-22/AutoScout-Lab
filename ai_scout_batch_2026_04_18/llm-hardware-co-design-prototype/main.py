```python
import abc
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration Constants ---
DEFAULT_PRECISION = 'FP32'
DEFAULT_BATCH_SIZE = 1
DEFAULT_DATA_SIZE = 1024 # Placeholder for tensor element count / input feature size
MIN_LATENCY_MS = 0.001 # Minimum simulated operation latency
MAX_ENERGY_MJ = 0.5 # Maximum simulated operation energy


# --- 1. Modular Hardware Abstraction (`hardware_models/`) ---

@dataclass
class SimulationResult:
    """Stores aggregated simulation metrics."""
    latency_ms: float = 0.0
    energy_mj: float = 0.0
    throughput_ops_per_ms: float = 0.0
    memory_bandwidth_gbs: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return (f"  Latency: {self.latency_ms:.4f} ms\n"
                f"  Energy: {self.energy_mj:.4f} mJ\n"
                f"  Throughput: {self.throughput_ops_per_ms:.2f} ops/ms\n"
                f"  Memory Bandwidth: {self.memory_bandwidth_gbs:.2f} GB/s")

class BaseHardwareModel(abc.ABC):
    """Abstract base class for specialized AI accelerator models."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def simulate_operation_cost(self, op_type: str, precision: str = DEFAULT_PRECISION,
                                data_size: int = DEFAULT_DATA_SIZE, is_sparse: bool = False,
                                batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[float, float, float]:
        """
        Simulates the latency, energy, and memory cost for a given operation.
        Returns (latency_ms, energy_mj, memory_access_gbs).
        """
        pass

    @abc.abstractmethod
    def supports_quantization(self, precision: str) -> bool:
        """Checks if the hardware model supports a given quantization precision."""
        pass

    @abc.abstractmethod
    def supports_sparse_ops(self) -> bool:
        """Checks if the hardware model has specialized support for sparse operations."""
        pass

    @abc.abstractmethod
    def get_supported_precisions(self) -> List[str]:
        """Returns a list of supported data precisions."""
        pass

class GenericGPUModel(BaseHardwareModel):
    """
    A generic GPU model for baseline comparison.
    Represents current typical LLM inference hardware.
    """
    def __init__(self):
        super().__init__("Generic GPU")
        self._supported_precisions = ['FP32', 'FP16', 'BF16', 'INT8']

    def simulate_operation_cost(self, op_type: str, precision: str = DEFAULT_PRECISION,
                                data_size: int = DEFAULT_DATA_SIZE, is_sparse: bool = False,
                                batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[float, float, float]:
        base_latency = data_size / 100000.0 * batch_size # Example: scales with data and batch
        base_energy = data_size / 50000.0 * batch_size # Example
        base_memory = data_size / 20000.0 * batch_size # Example

        # Adjust based on precision
        if precision == 'FP16' or precision == 'BF16':
            base_latency *= 0.7
            base_energy *= 0.6
            base_memory *= 0.8
        elif precision == 'INT8':
            base_latency *= 0.5
            base_energy *= 0.4
            base_memory *= 0.7

        # Adjust for sparse ops (some benefit, but not specialized)
        if is_sparse:
            base_latency *= 0.9 # Minor improvement
            base_energy *= 0.8 # Minor improvement
            base_memory *= 0.7 # Better memory utilization

        # Specific op types might have different costs
        if op_type == 'MATMUL':
            base_latency *= 1.2 # GPUs good at matmul, but maybe slightly higher base
        elif op_type == 'ATTENTION':
            base_latency *= 2.0 # More complex, higher cost
        elif op_type == 'ACTIVATION':
            base_latency *= 0.8 # Less complex
            base_energy *= 0.7

        return max(MIN_LATENCY_MS, base_latency), max(0.01, base_energy), max(0.01, base_memory)

    def supports_quantization(self, precision: str) -> bool:
        return precision in self._supported_precisions

    def supports_sparse_ops(self) -> bool:
        return True # General support, but not specialized acceleration

    def get_supported_precisions(self) -> List[str]:
        return self._supported_precisions

class PhotonicAccelerator(BaseHardwareModel):
    """
    Simulates a photonic computing accelerator, optimized for matrix multiplications
    and high-bandwidth data movement.
    """
    def __init__(self):
        super().__init__("Photonic Accelerator")
        self._supported_precisions = ['FP32', 'FP16', 'BF16', 'FP8'] # FP8 for specific photonic operations

    def simulate_operation_cost(self, op_type: str, precision: str = DEFAULT_PRECISION,
                                data_size: int = DEFAULT_DATA_SIZE, is_sparse: bool = False,
                                batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[float, float, float]:
        base_latency = data_size / 200000.0 * batch_size # Higher throughput
        base_energy = data_size / 80000.0 * batch_size # Lower energy for core operations
        base_memory = data_size / 10000.0 * batch_size # Potential higher internal bandwidth

        # Photonic excels at dense linear algebra
        if op_type == 'MATMUL' or op_type == 'ATTENTION':
            base_latency *= 0.3 # Significant speedup
            base_energy *= 0.2 # Energy efficiency
            base_memory *= 0.5 # High bandwidth utilization

        # Non-linear ops might have higher overhead due to O/E conversion
        if op_type == 'ACTIVATION':
            base_latency *= 1.5
            base_energy *= 1.8

        # Quantization benefits
        if precision == 'FP16' or precision == 'BF16':
            base_latency *= 0.8
            base_energy *= 0.7
        elif precision == 'FP8': # Very efficient for specific photonic architectures
            base_latency *= 0.5
            base_energy *= 0.4
        elif precision == 'INT8': # Generally not native to photonic, might have conversion overhead
            base_latency *= 1.5
            base_energy *= 1.2


        # Sparse operations might not be natively efficient unless specifically designed
        if is_sparse:
            base_latency *= 1.2 # Could be worse if not optimized
            base_energy *= 1.1

        return max(MIN_LATENCY_MS, base_latency), max(0.005, base_energy), max(0.05, base_memory)

    def supports_quantization(self, precision: str) -> bool:
        return precision in self._supported_precisions

    def supports_sparse_ops(self) -> bool:
        return False # Assuming no native sparse support for this basic model

    def get_supported_precisions(self) -> List[str]:
        return self._supported_precisions

class AnalogAIChip(BaseHardwareModel):
    """
    Simulates an analog AI chip, highly efficient for low-precision arithmetic
    and typically in-memory computation.
    """
    def __init__(self):
        super().__init__("Analog AI Chip")
        self._supported_precisions = ['FP16', 'BF16', 'INT8', 'FP8'] # Excels in low precision

    def simulate_operation_cost(self, op_type: str, precision: str = DEFAULT_PRECISION,
                                data_size: int = DEFAULT_DATA_SIZE, is_sparse: bool = False,
                                batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[float, float, float]:
        base_latency = data_size / 150000.0 * batch_size
        base_energy = data_size / 100000.0 * batch_size # Very energy efficient
        base_memory = data_size / 30000.0 * batch_size # In-memory computing

        # Best for low precision (FP8/INT8)
        if precision == 'FP32': # High overhead for higher precision
            base_latency *= 2.5
            base_energy *= 3.0
        elif precision == 'FP16' or precision == 'BF16':
            base_latency *= 0.8
            base_energy *= 0.7
        elif precision == 'INT8' or precision == 'FP8':
            base_latency *= 0.2 # Extremely fast
            base_energy *= 0.1 # Extremely energy efficient

        # Good for dense matrix operations at low precision
        if op_type == 'MATMUL' or op_type == 'ATTENTION':
            base_latency *= 0.5
            base_energy *= 0.4

        # Sparse operations might be less efficient if not specifically designed
        if is_sparse:
            base_latency *= 1.1
            base_energy *= 1.2

        return max(MIN_LATENCY_MS, base_latency), max(0.001, base_energy), max(0.02, base_memory)

    def supports_quantization(self, precision: str) -> bool:
        return precision in self._supported_precisions

    def supports_sparse_ops(self) -> bool:
        return False # Assuming no native sparse support for this basic model

    def get_supported_precisions(self) -> List[str]:
        return self._supported_precisions

class NeuromorphicChip(BaseHardwareModel):
    """
    Simulates a neuromorphic chip, specialized for sparse activations and
    event-driven computation, potentially high latency for dense, but very efficient for sparse.
    """
    def __init__(self):
        super().__init__("Neuromorphic Chip")
        self._supported_precisions = ['INT8', 'INT4'] # Often event-based, low precision

    def simulate_operation_cost(self, op_type: str, precision: str = DEFAULT_PRECISION,
                                data_size: int = DEFAULT_DATA_SIZE, is_sparse: bool = False,
                                batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[float, float, float]:
        base_latency = data_size / 50000.0 * batch_size # Generally higher for dense ops
        base_energy = data_size / 200000.0 * batch_size # Extremely low energy for sparse ops
        base_memory = data_size / 15000.0 * batch_size

        # Neuromorphic excels at sparse, event-driven ops
        if is_sparse and (op_type == 'MATMUL' or op_type == 'ATTENTION' or op_type == 'ACTIVATION'):
            base_latency *= 0.1 # Significant speedup for sparse
            base_energy *= 0.05 # Ultra low energy for sparse
            base_memory *= 0.1 # Very efficient memory for sparse patterns
        elif not is_sparse: # Poor performance for dense ops
            base_latency *= 5.0
            base_energy *= 3.0

        # Limited precision support, conversion overhead for higher precisions
        if precision == 'FP32' or precision == 'FP16' or precision == 'BF16':
            base_latency *= 4.0 # High conversion overhead
            base_energy *= 3.0
        elif precision == 'INT8' or precision == 'INT4':
            base_latency *= 0.5 # Native low precision
            base_energy *= 0.4

        return max(MIN_LATENCY_MS, base_latency), max(0.001, base_energy), max(0.01, base_memory)

    def supports_quantization(self, precision: str) -> bool:
        return precision in self._supported_precisions

    def supports_sparse_ops(self) -> bool:
        return True # Specialized for sparse operations

    def get_supported_precisions(self) -> List[str]:
        return self._supported_precisions


# --- 3. LLM Component Focus (`llm_components/`) ---

@dataclass
class Operation:
    """Represents a high-level operation in an LLM component graph."""
    op_type: str  # e.g., 'MATMUL', 'ADD', 'GELU', 'LAYERNORM', 'SOFTMAX', 'CONCAT'
    op_id: str    # Unique identifier for the operation
    inputs: List[str] = field(default_factory=list) # Placeholder for input tensor IDs
    outputs: List[str] = field(default_factory=list) # Placeholder for output tensor IDs
    data_size: int = DEFAULT_DATA_SIZE # Represents complexity/size of data involved
    is_sparse: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict) # For additional info like activation sparsity

@dataclass
class CompiledOperation:
    """Represents an operation after compiler optimization and hardware mapping."""
    original_op_id: str
    hardware_mapped_op: str # Could be same as op_type or a specific hardware instruction
    precision: str
    data_size: int
    is_sparse: bool
    estimated_cost: Tuple[float, float, float] # (latency_ms, energy_mj, memory_access_gbs)
    fused_ops: List[str] = field(default_factory=list) # List of original op_ids fused into this

def create_attention_block_graph(sequence_length: int = 128, hidden_size: int = 768) -> List[Operation]:
    """
    Creates a simplified computational graph for a single transformer attention block.
    This graph doesn't represent full data dependencies but a sequence of common operations.
    """
    graph: List[Operation] = []
    # Simplified Self-Attention: QKV projection -> MatMul (QK^T) -> Softmax -> MatMul (V)
    # Plus LayerNorm and FeedForward components.

    # LayerNorm 1
    graph.append(Operation('LAYERNORM', 'ln1', data_size=hidden_size*sequence_length))

    # QKV Projections (simplified as 3 separate MATMULs)
    graph.append(Operation('MATMUL', 'q_proj', inputs=['ln1_out'], outputs=['q'], data_size=hidden_size*hidden_size))
    graph.append(Operation('MATMUL', 'k_proj', inputs=['ln1_out'], outputs=['k'], data_size=hidden_size*hidden_size))
    graph.append(Operation('MATMUL', 'v_proj', inputs=['ln1_out'], outputs=['v'], data_size=hidden_size*hidden_size))

    # Attention Score Calculation (Q * K^T)
    graph.append(Operation('MATMUL', 'qk_matmul', inputs=['q', 'k'], outputs=['qk_scores'], data_size=sequence_length*sequence_length*hidden_size))

    # Softmax
    graph.append(Operation('SOFTMAX', 'softmax', inputs=['qk_scores'], outputs=['attention_weights'], data_size=sequence_length*sequence_length))

    # Weighted Sum (Attention Weights * V)
    graph.append(Operation('MATMUL', 'weighted_v_matmul', inputs=['attention_weights', 'v'], outputs=['attn_output'], data_size=sequence_length*hidden_size))

    # Output Projection (Linear layer)
    graph.append(Operation('MATMUL', 'attn_out_proj', inputs=['attn_output'], outputs=['attn_block_out'], data_size=hidden_size*hidden_size))

    # Add & Norm (Residual connection + LayerNorm 2)
    # For simplicity, we model residual as an ADD and then another LayerNorm
    graph.append(Operation('ADD', 'residual_add', inputs=['original_input', 'attn_block_out'], outputs=['add_out'], data_size=hidden_size*sequence_length))
    graph.append(Operation('LAYERNORM', 'ln2', inputs=['add_out'], outputs=['ln2_out'], data_size=hidden_size*sequence_length))

    # Feed-Forward Network (2 Linear layers with GELU)
    # Can introduce sparsity here for demonstration
    is_ffn_sparse = random.random() < 0.3 # 30% chance FFN is sparse
    graph.append(Operation('MATMUL', 'ffn_in_proj', inputs=['ln2_out'], outputs=['ffn_intermediate'], data_size=hidden_size*hidden_size*4, is_sparse=is_ffn_sparse, metadata={'sparsity': 0.7 if is_ffn_sparse else 0.0}))
    graph.append(Operation('GELU', 'gelu', inputs=['ffn_intermediate'], outputs=['gelu_out'], data_size=hidden_size*hidden_size*4))
    graph.append(Operation('MATMUL', 'ffn_out_proj', inputs=['gelu_out'], outputs=['ffn_output'], data_size=hidden_size*hidden_size))

    # Final Add & Norm
    graph.append(Operation('ADD', 'final_residual_add', inputs=['ln2_out', 'ffn_output'], outputs=['final_add_out'], data_size=hidden_size*sequence_length))
    graph.append(Operation('LAYERNORM', 'final_ln', inputs=['final_add_out'], outputs=['block_final_output'], data_size=hidden_size*sequence_length))

    return graph


# --- 2. Adaptive Software Stack (`software_stack/`) ---

class Quantizer:
    """Provides software implementations for various low-precision arithmetic."""
    def __init__(self):
        pass

    def apply_quantization(self, graph: List[Operation], target_precision: str,
                           hardware_model: BaseHardwareModel) -> List[Operation]:
        """
        Applies a quantization scheme to the graph based on hardware capabilities.
        Returns a new graph with updated operation precisions.
        """
        if not hardware_model.supports_quantization(target_precision):
            print(f"  WARNING: Hardware '{hardware_model.name}' does not natively support "
                  f"'{target_precision}'. Falling back to best supported or default.")
            supported = hardware_model.get_supported_precisions()
            if supported:
                actual_precision = supported[0] # Fallback to first supported
            else:
                actual_precision = DEFAULT_PRECISION
            print(f"  Using '{actual_precision}' instead for quantization attempt.")
        else:
            actual_precision = target_precision

        quantized_graph = []
        for op in graph:
            # In a real scenario, this would involve more complex analysis
            # like range estimation, calibration, and per-layer quantization.
            # For this prototype, we simply mark the operation for the target precision.
            new_op = Operation(
                op_type=op.op_type,
                op_id=op.op_id,
                inputs=op.inputs,
                outputs=op.outputs,
                data_size=op.data_size,
                is_sparse=op.is_sparse,
                metadata={**op.metadata, 'precision': actual_precision}
            )
            quantized_graph.append(new_op)
        return quantized_graph

class HardwareAwareCompiler:
    """
    Takes an LLM component's computational graph and performs hardware-aware optimizations.
    Generates 'code' (sequences of operations and data movements) tailored for the chosen hardware model.
    """
    def __init__(self):
        pass

    def compile(self, llm_graph: List[Operation], hardware_model: BaseHardwareModel,
                target_precision: str = DEFAULT_PRECISION) -> List[CompiledOperation]:
        """
        Compiles the LLM graph for the given hardware model and target precision.
        """
        compiled_ops: List[CompiledOperation] = []
        print(f"  Compiler: Optimizing for {hardware_model.name} with {target_precision}...")

        # --- Placeholder for hardware-aware optimizations ---
        # 1. Operator Fusion: Combine compatible operations
        #    Example: LayerNorm might fuse Scale+Bias+Normalize. MatMul + Bias + Activation.
        fused_ops_map = {} # Maps original op_id to a potentially fused op
        # Dummy fusion for demonstration
        fused_graph_representation = []
        i = 0
        while i < len(llm_graph):
            current_op = llm_graph[i]
            current_precision = current_op.metadata.get('precision', target_precision)

            # Example: Fuse MatMul and following Activation
            if current_op.op_type == 'MATMUL' and i + 1 < len(llm_graph) and \
               llm_graph[i+1].op_type in ['GELU', 'RELU'] and \
               current_op.outputs[0] == llm_graph[i+1].inputs[0]:
                print(f"    Compiler: Fusing {current_op.op_type} and {llm_graph[i+1].op_type}")
                # Create a "fused" op that combines their characteristics
                fused_op_type = f"{current_op.op_type}_{llm_graph[i+1].op_type}"
                # For simulation, we'll use the MatMul's data_size, and adjust cost
                # A real compiler would analyze resource usage more deeply.
                fused_op = Operation(
                    op_type=fused_op_type,
                    op_id=f"fused_{current_op.op_id}_{llm_graph[i+1].op_id}",
                    inputs=current_op.inputs,
                    outputs=llm_graph[i+1].outputs,
                    data_size=current_op.data_size, # Use larger data size
                    is_sparse=current_op.is_sparse or llm_graph[i+1].is_sparse,
                    metadata={'fused_components': [current_op.op_id, llm_graph[i+1].op_id],
                              'precision': current_precision}
                )
                fused_graph_representation.append(fused_op)
                fused_ops_map[current_op.op_id] = fused_op.op_id
                fused_ops_map[llm_graph[i+1].op_id] = fused_op.op_id
                i += 2 # Skip next op as it's fused
            else:
                fused_graph_representation.append(current_op)
                fused_ops_map[current_op.op_id] = current_op.op_id
                i += 1

        # 2. Reordering, Data Layout Transformations, Mapping to Specialized Units
        #    For this prototype, we'll process sequentially but acknowledge these steps.
        #    The `simulate_operation_cost` implicitly handles 'mapping to specialized units'
        #    by having different costs for different op_types on different hardware.

        for op in fused_graph_representation:
            current_precision = op.metadata.get('precision', target_precision)
            # Ensure the precision is supported by the hardware, fallback if not
            if not hardware_model.supports_quantization(current_precision):
                supported = hardware_model.get_supported_precisions()
                if current_precision != DEFAULT_PRECISION and supported and DEFAULT_PRECISION not in supported:
                    actual_precision = supported[0]
                else: # Fallback to default if target not supported
                    actual_precision = DEFAULT_PRECISION if hardware_model.supports_quantization(DEFAULT_PRECISION) else (supported[0] if supported else 'Unknown')

                print(f"    Compiler: Precision '{current_precision}' not fully supported by {hardware_model.name} for op '{op.op_id}'. "
                      f"Using '{actual_precision}'.")
            else:
                actual_precision = current_precision

            # Estimate cost on the hardware model
            try:
                latency, energy, memory_access = hardware_model.simulate_operation_cost(
                    op_type=op.op_type,
                    precision=actual_precision,
                    data_size=op.data_size,
                    is_sparse=op.is_sparse and hardware_model.supports_sparse_ops(), # Only leverage sparse if hardware supports
                    batch_size=DEFAULT_BATCH_SIZE # Batch size handled by engine, but for per-op cost, assume 1
                )
            except Exception as e:
                print(f"    ERROR: Failed to simulate cost for op '{op.op_id}' on {hardware_model.name}: {e}")
                latency, energy, memory_access = 1000.0, 1000.0, 100.0 # High penalty

            compiled_op = CompiledOperation(
                original_op_id=op.op_id, # Could be the fused ID
                hardware_mapped_op=op.op_type, # Simplification, could be e.g. 'PhotonicMatMul'
                precision=actual_precision,
                data_size=op.data_size,
                is_sparse=op.is_sparse and hardware_model.supports_sparse_ops(),
                estimated_cost=(latency, energy, memory_access),
                fused_ops=op.metadata.get('fused_components', [])
            )
            compiled_ops.append(compiled_op)

        return compiled_ops

class InferenceEngine:
    """
    Executes the compiled graph on the simulated hardware model.
    Interacts with hardware models to query simulated latencies and energy,
    handles dynamic batching, and manages data flow.
    """
    def __init__(self):
        self._total_latency = 0.0
        self._total_energy = 0.0
        self._total_memory_access = 0.0
        self._operations_executed = 0

    def execute(self, compiled_graph: List[CompiledOperation], hardware_model: BaseHardwareModel,
                batch_size: int = DEFAULT_BATCH_SIZE) -> SimulationResult:
        """
        Executes the compiled graph and returns simulated performance metrics.
        """
        print(f"  Inference Engine: Executing compiled graph on {hardware_model.name} (Batch Size: {batch_size})...")
        total_latency_ms = 0.0
        total_energy_mj = 0.0
        total_memory_access_gbs = 0.0
        operations_count = 0

        # Placeholder for dynamic batching:
        # In a real engine, batch size would affect memory layout,
        # communication, and potentially operator scheduling.
        # For simulation, we scale the per-op costs by batch_size.

        for compiled_op in compiled_graph:
            latency, energy, memory_access = compiled_op.estimated_cost
            # Scale costs by batch size for overall execution
            total_latency_ms += latency * batch_size
            total_energy_mj += energy * batch_size
            total_memory_access_gbs += memory_access * batch_size
            operations_count += 1
            # print(f"    Executing op '{compiled_op.original_op_id}' ({compiled_op.hardware_mapped_op}) - "
            #       f"Lat: {latency*batch_size:.3f}ms, Energy: {energy*batch_size:.3f}mJ")

        if operations_count > 0:
            avg_op_latency = total_latency_ms / operations_count
            throughput_ops_per_ms = operations_count / total_latency_ms if total_latency_ms > 0 else float('inf')
        else:
            avg_op_latency = 0.0
            throughput_ops_per_ms = 0.0

        print(f"  Inference Engine: Execution finished. Total Latency: {total_latency_ms:.4f} ms")

        # Basic detail tracking
        details = {
            "hardware_model": hardware_model.name,
            "batch_size": batch_size,
            "operations_executed": operations_count
        }

        return SimulationResult(
            latency_ms=total_latency_ms,
            energy_mj=total_energy_mj,
            throughput_ops_per_ms=throughput_ops_per_ms,
            memory_bandwidth_gbs=total_memory_access_gbs, # This is a sum, not bandwidth, need refinement
            details=details
        )


# --- 4. Performance and Energy Simulation (`benchmarking/`) ---

class BenchmarkRunner:
    """
    The core orchestrator for running simulations and collecting metrics.
    Compares co-designed stack against a generic 'existing hardware' model.
    """
    def __init__(self, quantizer: Quantizer, compiler: HardwareAwareCompiler, inference_engine: InferenceEngine):
        self._quantizer = quantizer
        self._compiler = compiler
        self._inference_engine = inference_engine

    def run_benchmark(self, llm_graph: List[Operation], hardware_model: BaseHardwareModel,
                      precision: str = DEFAULT_PRECISION, batch_size: int = DEFAULT_BATCH_SIZE) -> SimulationResult:
        """
        Runs a full benchmark simulation for a given hardware and software configuration.
        """
        print(f"\n--- Running Benchmark for '{hardware_model.name}' (Precision: {precision}, Batch: {batch_size}) ---")

        # 1. Apply quantization (if requested and supported)
        quantized_graph = self._quantizer.apply_quantization(llm_graph, precision, hardware_model)
        print(f"  Quantization applied. Operations now targeting '{precision}'.")

        # 2. Compile the graph
        compiled_graph = self._compiler.compile(quantized_graph, hardware_model, precision)
        print(f"  Graph compiled with {len(compiled_graph)} final operations.")

        # 3. Execute the compiled graph
        simulation_result = self._inference_engine.execute(compiled_graph, hardware_model, batch_size)

        return simulation_result

# --- Main Execution Logic ---
if __name__ == "__main__":
    print("--- LLM Hardware-Software Co-design Prototype ---")

    # Initialize components
    quantizer = Quantizer()
    compiler = HardwareAwareCompiler()
    inference_engine = InferenceEngine()
    benchmark_runner = BenchmarkRunner(quantizer, compiler, inference_engine)

    # Define LLM component (e.g., a single attention block)
    print("\n--- Creating LLM Component Graph (Attention Block) ---")
    llm_component_graph = create_attention_block_graph(sequence_length=512, hidden_size=1024)
    print(f"Created a graph with {len(llm_component_graph)} operations.")

    # Define various hardware models
    hardware_models: List[BaseHardwareModel] = [
        GenericGPUModel(),
        PhotonicAccelerator(),
        AnalogAIChip(),
        NeuromorphicChip()
    ]

    # --- Run Benchmarks for Different Scenarios ---
    results: Dict[str, SimulationResult] = {}
    baseline_result: Optional[SimulationResult] = None

    # Scenario 1: Baseline (Generic GPU, FP32)
    baseline_result = benchmark_runner.run_benchmark(
        llm_component_graph, hardware_models[0], precision='FP32', batch_size=DEFAULT_BATCH_SIZE
    )
    results[f"{hardware_models[0].name}_FP32"] = baseline_result

    # Scenario 2: Generic GPU, INT8 (Quantized)
    gpu_int8_result = benchmark_runner.run_benchmark(
        llm_component_graph, hardware_models[0], precision='INT8', batch_size=DEFAULT_BATCH_SIZE
    )
    results[f"{hardware_models[0].name}_INT8"] = gpu_int8_result

    # Scenario 3: Photonic Accelerator, FP16
    photonic_fp16_result = benchmark_runner.run_benchmark(
        llm_component_graph, hardware_models[1], precision='FP16', batch_size=DEFAULT_BATCH_SIZE
    )
    results[f"{hardware_models[1].name}_FP16"] = photonic_fp16_result

    # Scenario 4: Analog AI Chip, FP8
    analog_fp8_result = benchmark_runner.run_benchmark(
        llm_component_graph, hardware_models[2], precision='FP8', batch_size=DEFAULT_BATCH_SIZE
    )
    results[f"{hardware_models[2].name}_FP8"] = analog_fp8_result

    # Scenario 5: Neuromorphic Chip, INT8 (with potential sparse operations)
    # Re-generate graph for neuromorphic to potentially have more sparse ops
    print("\n--- Creating LLM Component Graph (Attention Block) with higher sparsity potential for Neuromorphic ---")
    sparse_llm_graph = create_attention_block_graph(sequence_length=512, hidden_size=1024)
    for op in sparse_llm_graph:
        if 'MATMUL' in op.op_type and random.random() < 0.6: # Higher chance of sparsity for this test
            op.is_sparse = True
            op.metadata['sparsity'] = 0.8
    neuromorphic_int8_result = benchmark_runner.run_benchmark(
        sparse_llm_graph, hardware_models[3], precision='INT8', batch_size=DEFAULT_BATCH_SIZE
    )
    results[f"{hardware_models[3].name}_INT8"] = neuromorphic_int8_result

    # Scenario 6: Photonic Accelerator, FP32 (higher batch size)
    photonic_fp32_batch4_result = benchmark_runner.run_benchmark(
        llm_component_graph, hardware_models[1], precision='FP32', batch_size=4
    )
    results[f"{hardware_models[1].name}_FP32_Batch4"] = photonic_fp32_batch4_result

    # --- Print Comparison Summary ---
    print("\n\n--- Benchmark Summary ---")
    if baseline_result:
        print(f"\nBaseline ({hardware_models[0].name}, FP32, Batch {DEFAULT_BATCH_SIZE}):")
        print(baseline_result)

        for name, res in results.items():
            if name != f"{hardware_models[0].name}_FP32":
                print(f"\nScenario: {name}")
                print(res)
                latency_speedup = baseline_result.latency_ms / res.latency_ms if res.latency_ms > 0 else float('inf')
                energy_savings = (baseline_result.energy_mj - res.energy_mj) / baseline_result.energy_mj if baseline_result.energy_mj > 0 else 0
                print(f"  Latency Speedup vs. Baseline: {latency_speedup:.2f}x")
                print(f"  Energy Savings vs. Baseline: {energy_savings:.2%}")
    else:
        print("No baseline result available.")

    print("\n--- Prototype Simulation Complete ---")

```