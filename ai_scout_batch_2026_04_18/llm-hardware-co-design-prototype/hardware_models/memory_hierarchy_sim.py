```python
import math

class MemoryLevel:
    """
    Represents a single level within the memory hierarchy (e.g., L1 Cache, HBM, DDR).
    Each level has distinct characteristics influencing access latency and energy.
    """
    def __init__(self,
                 name: str,
                 size_bytes: int,
                 access_latency_ns: float,
                 bandwidth_gbps: float,
                 read_energy_pj_per_byte: float,
                 write_energy_pj_per_byte: float):
        """
        Initializes a MemoryLevel with its performance and energy parameters.

        Args:
            name (str): The name of the memory level (e.g., "L1 Cache", "HBM2", "DDR5").
            size_bytes (int): The total capacity of this memory level in bytes.
            access_latency_ns (float): The base latency for a minimal access to this level, in nanoseconds.
                                      This typically represents the latency to start a transfer.
            bandwidth_gbps (float): The sustained data transfer rate of this level, in gigabytes per second (GBps).
            read_energy_pj_per_byte (float): Energy consumed per byte for read operations, in picojoules (pJ).
            write_energy_pj_per_byte (float): Energy consumed per byte for write operations, in picojoules (pJ).

        Raises:
            ValueError: If any parameter is invalid (e.g., non-positive numerical values, empty name).
            TypeError: If parameters are of incorrect types.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Memory level name must be a non-empty string.")
        
        # Validate numerical parameters
        numerical_params = {
            "size_bytes": size_bytes,
            "access_latency_ns": access_latency_ns,
            "bandwidth_gbps": bandwidth_gbps,
            "read_energy_pj_per_byte": read_energy_pj_per_byte,
            "write_energy_pj_per_byte": write_energy_pj_per_byte
        }
        for param_name, value in numerical_params.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"{param_name} must be an integer or float.")
            if value < 0:
                raise ValueError(f"{param_name} must be non-negative.")
        
        if bandwidth_gbps == 0 and size_bytes > 0: # A memory level with capacity must have bandwidth
             raise ValueError("Bandwidth cannot be zero for a memory level with non-zero size.")


        self.name = name
        self.size_bytes = size_bytes
        self.access_latency_ns = access_latency_ns
        self.bandwidth_gbps = bandwidth_gbps
        self.read_energy_pj_per_byte = read_energy_pj_per_byte
        self.write_energy_pj_per_byte = write_energy_pj_per_byte

    @property
    def bandwidth_bytes_per_ns(self) -> float:
        """Converts bandwidth from GBps to Bytes/ns for easier calculation."""
        # 1 GB/s = 1e9 Bytes/s
        # 1 ns = 1e-9 s
        # GBps * (1e9 Bytes / 1 GB) * (1e-9 s / 1 ns) = Bytes/ns
        return self.bandwidth_gbps 

    def __repr__(self):
        return (f"MemoryLevel(name='{self.name}', size={self.size_bytes / (1024**2):.2f}MB, "
                f"latency={self.access_latency_ns:.2f}ns, bandwidth={self.bandwidth_gbps:.2f}GBps, "
                f"read_energy={self.read_energy_pj_per_byte:.2f}pJ/B, write_energy={self.write_energy_pj_per_byte:.2f}pJ/B)")


class MemoryHierarchy:
    """
    Simulates memory access patterns across a defined memory hierarchy.
    The hierarchy is assumed to be ordered from fastest/closest (e.g., L1 cache)
    to slowest/farthest (e.g., DDR).
    """
    def __init__(self, memory_levels: list[MemoryLevel]):
        """
        Initializes the MemoryHierarchy with a list of MemoryLevel objects.

        Args:
            memory_levels (list[MemoryLevel]): A list of MemoryLevel objects,
                                               ordered from fastest to slowest.

        Raises:
            TypeError: If memory_levels is not a list of MemoryLevel objects.
            ValueError: If memory_levels is empty.
        """
        if not isinstance(memory_levels, list):
            raise TypeError("memory_levels must be a list.")
        if not all(isinstance(level, MemoryLevel) for level in memory_levels):
            raise TypeError("All elements in memory_levels must be MemoryLevel objects.")
        if not memory_levels:
            raise ValueError("Memory hierarchy must contain at least one memory level.")

        # Sort levels by increasing latency to ensure consistency,
        # although users are expected to provide them in order.
        self.memory_levels = sorted(memory_levels, key=lambda level: level.access_latency_ns)

    def simulate_access(self, data_size_bytes: int, is_write: bool = False) -> tuple[float, float]:
        """
        Simulates a memory access for a given data size, returning estimated latency and energy consumption.

        This simulation employs a simplified model for a functional prototype:
        It determines the 'effective' memory level from which the entire requested
        `data_size_bytes` would be primarily served. This is the fastest memory level
        that can fully contain the data. If the data size exceeds the capacity of all
        defined levels, it defaults to using the characteristics of the largest (slowest)
        level provided.

        The total latency is calculated as the sum of the base access latency of the
        effective level and the time required to transfer `data_size_bytes` at that
        level's bandwidth. Energy is calculated based on the data size and the
        effective level's per-byte energy cost.

        Args:
            data_size_bytes (int): The size of the data block to access, in bytes.
            is_write (bool): True if this is a write operation, False for a read.

        Returns:
            tuple[float, float]: A tuple containing (total_latency_ns, total_energy_pj).
                                 Returns (0.0, 0.0) if data_size_bytes is 0.

        Raises:
            ValueError: If data_size_bytes is negative.
        """
        if data_size_bytes < 0:
            raise ValueError("data_size_bytes must be non-negative.")
        if data_size_bytes == 0:
            return 0.0, 0.0

        # Default to the slowest/largest level if data exceeds all capacities
        effective_level: MemoryLevel = self.memory_levels[-1] 

        # Find the fastest memory level that can contain the entire data_size_bytes
        # This simulates that if data fits, it would be served from that level
        for level in self.memory_levels:
            if data_size_bytes <= level.size_bytes:
                effective_level = level
                break
        
        # Calculate transfer latency based on the effective level's bandwidth
        # bandwidth_bytes_per_ns is GBps, which is also Bytes/ns directly
        # Example: 100 GBps = 100 Bytes/ns
        if effective_level.bandwidth_bytes_per_ns == 0:
            # This case should ideally be caught during MemoryLevel init if size > 0
            # but as a fallback, prevent division by zero
            transfer_latency_ns = float('inf') 
        else:
            transfer_latency_ns = data_size_bytes / effective_level.bandwidth_bytes_per_ns

        # Total latency is base access latency + data transfer latency
        total_latency_ns = effective_level.access_latency_ns + transfer_latency_ns

        # Calculate energy consumption
        if is_write:
            total_energy_pj = data_size_bytes * effective_level.write_energy_pj_per_byte
        else:
            total_energy_pj = data_size_bytes * effective_level.read_energy_pj_per_byte

        return total_latency_ns, total_energy_pj

```