```python
import os

class Config:
    """
    Global configuration settings for the LLM Hardware Co-Design Prototype.

    This class defines parameters for simulating novel hardware architectures,
    optimizing the software stack, specifying LLM components for focus,
    and configuring benchmarking and performance simulation.
    """

    # --- Project General Settings ---
    PROJECT_NAME = "LLM Hardware Co-Design Prototype"
    # Root directory for all simulation outputs, logs, and reports.
    # Automatically created if it doesn't exist.
    OUTPUT_DIR = os.path.join(os.getcwd(), "simulation_results")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True) # exist_ok=True prevents error if dir already exists

    # Default hardware model to use for simulations if not explicitly specified.
    DEFAULT_HARDWARE_MODEL = "PHOTONIC_ACCELERATOR"

    # --- Hardware Model Definitions ---
    # A dictionary defining various specialized AI accelerator models.
    # Each model includes simulated characteristics relevant to performance, energy,
    # memory, and feature support (e.g., quantization, sparsity).
    HARDWARE_MODELS = {
        "PHOTONIC_ACCELERATOR": {
            "description": "A novel photonic computing architecture leveraging light for computation.",
            "type": "photonic",
            "latency_factor_scalar_ops": 0.8,  # Relative speed compared to generic baseline (lower is faster)
            "latency_factor_matrix_ops": 0.6,  # Photonic excels in parallel matrix operations
            "energy_per_op_joules_avg": 0.5e-12, # Estimated average energy per generic operation
            "memory_bandwidth_gbps": 5000,     # Very high potential bandwidth due to co-integration
            "max_memory_gb": 128,              # Total usable memory
            "supported_quantization": ["FP32", "FP16", "BF16", "FP8_E5M2", "INT8"], # Supported data types
            "sparse_compute_efficiency_factor": 0.4, # Highly efficient for sparse computations
            "attention_mechanism_efficiency_factor": 0.5, # Specialized hardware for attention patterns
            "power_draw_watts_idle": 50,       # Estimated idle power consumption
            "power_draw_watts_peak": 300,      # Estimated peak power consumption
        },
        "ANALOG_COMPUTE_CHIP": {
            "description": "An analog AI chip leveraging in-memory computing principles.",
            "type": "analog",
            "latency_factor_scalar_ops": 1.2,
            "latency_factor_matrix_ops": 0.7,
            "energy_per_op_joules_avg": 0.2e-12, # Extremely low energy, potentially at cost of higher latency base
            "memory_bandwidth_gbps": 3000,
            "max_memory_gb": 64,
            "supported_quantization": ["INT8", "FP8_E5M2", "FP8_E4M3"], # Often focused on lower precision
            "sparse_compute_efficiency_factor": 0.6,
            "attention_mechanism_efficiency_factor": 0.7,
            "power_draw_watts_idle": 30,
            "power_draw_watts_peak": 200,
        },
        "NEUROMORPHIC_CHIP": {
            "description": "A neuromorphic processor optimized for sparse, event-driven computation.",
            "type": "neuromorphic",
            "latency_factor_scalar_ops": 1.5,
            "latency_factor_matrix_ops": 0.9,
            "energy_per_op_joules_avg": 0.1e-12, # Extremely low energy for sparse operations
            "memory_bandwidth_gbps": 1000,
            "max_memory_gb": 32,
            "supported_quantization": ["INT4", "INT8", "BINARY"], # Very low precision focus
            "sparse_compute_efficiency_factor": 0.1, # Extremely efficient for sparse activations
            "attention_mechanism_efficiency_factor": 0.8, # Can be adapted, but less direct fit than photonic
            "power_draw_watts_idle": 10,
            "power_draw_watts_peak": 100,
        },
        "GENERIC_GPU_BASELINE": {
            "description": "A high-level model representing a contemporary GPU for baseline comparison.",
            "type": "gpu",
            "latency_factor_scalar_ops": 1.0,  # Baseline for comparison
            "latency_factor_matrix_ops": 1.0,
            "energy_per_op_joules_avg": 1.0e-12, # Baseline energy
            "memory_bandwidth_gbps": 1500,
            "max_memory_gb": 80,
            "supported_quantization": ["FP32", "FP16", "BF16", "INT8"],
            "sparse_compute_efficiency_factor": 0.9, # Some sparsity support, but not native hardware
            "attention_mechanism_efficiency_factor": 1.0, # Standard implementation performance
            "power_draw_watts_idle": 80,
            "power_draw_watts_peak": 450,
        },
    }

    # --- Software Stack Configuration ---
    # Settings for the compiler, inference engine, and quantization libraries.
    SOFTWARE_STACK = {
        "COMPILER": {
            "OPTIMIZATION_LEVEL": "O2", # 'O0': No optimizations, 'O1': Basic, 'O2': Aggressive, 'O3': Max
            "ENABLE_OPERATOR_FUSION": True, # Fuse multiple small operations into larger kernels
            "ENABLE_DATA_LAYOUT_TRANSFORMATIONS": True, # Optimize data access patterns for hardware
            "ENABLE_HARDWARE_SPECIFIC_INTRINSICS": True, # Utilize specialized hardware instructions
            "TARGET_QUANTIZATION_PRECISION": None, # If None, compiler determines best precision based on hardware
        },
        "INFERENCE_ENGINE": {
            "SCHEDULER_TYPE": "DYNAMIC_BATCHING_AWARE", # Options: 'FIFO', 'PRIORITY', 'DYNAMIC_BATCHING_AWARE'
            "DYNAMIC_BATCHING_ENABLED": True, # Enable dynamic batching for variable workloads
            "MAX_BATCH_SIZE": 16, # Maximum number of requests/tokens to batch together
            "BATCH_TIMEOUT_MS": 10, # Max time in ms to wait for more requests before processing current batch
            "MEMORY_MANAGEMENT_STRATEGY": "OPTIMIZED_PAGE_ALLOCATION", # How memory is allocated and reused
        },
        "QUANTIZATION_LIBRARIES": {
            "DEFAULT_QUANTIZATION_SCHEME": "INT8_SYMMETRIC", # E.g., 'INT8_SYMMETRIC', 'FP8_E5M2_SAFETENSORS'
            "SUPPORTED_SOFTWARE_PRECISIONS": ["FP32", "FP16", "BF16", "INT8", "FP8_E5M2", "FP8_E4M3", "INT4", "BINARY"],
            "QUANTIZATION_AWARE_TRAINING_ENABLED": False, # Simulate the impact of QAT during optimization
            "CALIBRATION_DATASET_SIZE": 128, # Number of samples for post-training quantization calibration
        },
    }

    # --- LLM Component Focus ---
    # Specifies which critical LLM components will be focused on for simulation
    # and their typical input parameters.
    LLM_COMPONENTS = {
        "TARGET_COMPONENTS": [
            "ATTENTION_BLOCK",
            "SPARSE_FEED_FORWARD_LAYER",
            "GELU_ACTIVATION",
            # Additional components can be added here as the prototype evolves
        ],
        "COMPONENT_INPUT_SHAPES": {
            "ATTENTION_BLOCK": {
                "input_tensor_shape": (1, 128, 768),  # (batch_size, sequence_length, hidden_dim)
                "num_heads": 12,
                "head_dim": 64,
                "is_causal": True,
            },
            "SPARSE_FEED_FORWARD_LAYER": {
                "input_tensor_shape": (1, 128, 768),
                "hidden_dim": 3072, # Typically 4x hidden_dim
                "sparsity_ratio": 0.7, # Percentage of activations that are zero
            },
            "GELU_ACTIVATION": {
                "input_tensor_shape": (1, 128, 3072),
            },
            # Example for another component:
            # "MULTIHEAD_ATTENTION": {
            #     "input_tensor_shape": (1, 1024, 1024),
            #     "num_heads": 16,
            #     "head_dim": 64,
            # }
        },
        "LLM_MODEL_CONTEXT_PARAMS": { # General parameters that might influence component behavior
            "HIDDEN_DIM": 768,
            "NUM_LAYERS": 24,
            "SEQUENCE_LENGTH_MAX": 2048,
            "VOCAB_SIZE": 50257,
        }
    }

    # --- Benchmarking and Simulation Settings ---
    # Parameters for running simulations, collecting metrics, and reporting results.
    BENCHMARKING = {
        "NUM_SIMULATION_RUNS": 5, # Number of times to run each benchmark for robust averaging
        "WARMUP_RUNS": 1,         # Number of initial runs to discard (e.g., for JIT compilation, cache priming)
        "BASELINE_HARDWARE_MODEL": "GENERIC_GPU_BASELINE", # The hardware model to compare against
        "REPORT_FORMAT": "CSV",   # Options: 'CSV', 'JSON', 'TABLE', 'CONSOLE'
        "METRICS_TO_COLLECT": [
            "latency_ms",          # Total execution time in milliseconds
            "throughput_tokens_s", # Tokens processed per second
            "energy_joules",       # Total energy consumed in Joules
            "memory_usage_gb",     # Peak memory usage in Gigabytes
            "flops_per_watt",      # Computational efficiency metric
            "total_power_watts_avg", # Average power consumption during execution
            "memory_bandwidth_utilization_percent", # Percentage of max bandwidth used
        ],
        "LOG_LEVEL": "INFO",      # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
        "ENABLE_DETAILED_PROFILING": False, # If True, enables fine-grained operation-level metrics collection
    }

    # --- Miscellaneous Settings ---
    DEBUG_MODE = False # Enable/disable general debugging features
    LOG_FILE = os.path.join(OUTPUT_DIR, "prototype_log.log") # Path for the main log file

    @classmethod
    def get_hardware_config(cls, model_name: str) -> dict:
        """
        Retrieves the configuration dictionary for a specific hardware model.

        Args:
            model_name (str): The name of the hardware model (key in HARDWARE_MODELS).

        Returns:
            dict: The configuration dictionary for the specified hardware model.

        Raises:
            ValueError: If the specified hardware model name is not found.
        """
        config = cls.HARDWARE_MODELS.get(model_name)
        if config is None:
            raise ValueError(
                f"Hardware model '{model_name}' not found in Config.HARDWARE_MODELS. "
                f"Available models are: {list(cls.HARDWARE_MODELS.keys())}"
            )
        return config

    @classmethod
    def get_component_input_shape(cls, component_name: str) -> dict:
        """
        Retrieves the default input shape and associated parameters for a specific LLM component.

        Args:
            component_name (str): The name of the LLM component (key in COMPONENT_INPUT_SHAPES).

        Returns:
            dict: A dictionary containing input tensor shape and other relevant parameters.

        Raises:
            ValueError: If the specified LLM component name is not found.
        """
        shape_params = cls.LLM_COMPONENTS["COMPONENT_INPUT_SHAPES"].get(component_name)
        if shape_params is None:
            raise ValueError(
                f"Input shape parameters for LLM component '{component_name}' not defined in config. "
                f"Available components are: {list(cls.LLM_COMPONENTS['COMPONENT_INPUT_SHAPES'].keys())}"
            )
        return shape_params
```