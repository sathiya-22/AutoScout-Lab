```python
import threading
from typing import Dict, Union, Any

class ResourceManager:
    """
    Manages the allocation and tracking of resources (e.g., LLM tokens, compute, memory)
    across multiple agents in a multi-agent AI system.

    Ensures fair usage, prevents bottlenecks, and optimizes cost, especially in a
    distributed setup. This prototype uses in-memory tracking with basic locking.
    For a truly distributed system, the underlying storage and locking mechanisms
    would be replaced by distributed equivalents (e.g., Redis, etcd, Zookeeper).
    """

    def __init__(self, initial_resources: Dict[str, Union[int, float]]):
        """
        Initializes the ResourceManager with the total available resources.

        Args:
            initial_resources (Dict[str, Union[int, float]]): A dictionary mapping
                                                             resource names (str)
                                                             to their total available
                                                             quantities (int or float).
                                                             Example: {"LLM_TOKENS": 1_000_000,
                                                                       "CPU_CORES": 16,
                                                                       "MEMORY_MB": 32768}
        """
        if not isinstance(initial_resources, dict) or not all(
            isinstance(k, str) and isinstance(v, (int, float)) and v >= 0
            for k, v in initial_resources.items()
        ):
            raise ValueError("initial_resources must be a dictionary of string keys and non-negative numeric values.")

        self._total_resources: Dict[str, Union[int, float]] = initial_resources.copy()
        self._available_resources: Dict[str, Union[int, float]] = initial_resources.copy()
        # _allocated_resources: Stores resources allocated to each agent.
        # Structure: {agent_id: {resource_type: quantity}}
        self._allocated_resources: Dict[str, Dict[str, Union[int, float]]] = {}
        self._lock = threading.Lock() # For local concurrency control

    def _validate_resource_request(self, resources: Dict[str, Union[int, float]]) -> None:
        """Helper to validate resource dictionary format and values."""
        if not isinstance(resources, dict):
            raise TypeError("Resource request/release must be a dictionary.")
        for res_type, quantity in resources.items():
            if not isinstance(res_type, str):
                raise TypeError(f"Resource type '{res_type}' must be a string.")
            if not isinstance(quantity, (int, float)):
                raise TypeError(f"Quantity for resource '{res_type}' must be an int or float.")
            if quantity < 0:
                raise ValueError(f"Quantity for resource '{res_type}' cannot be negative.")

    def register_agent(self, agent_id: str) -> None:
        """
        Registers an agent with the resource manager.
        This initializes its allocation tracking.

        Args:
            agent_id (str): The unique identifier of the agent.
        """
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("Agent ID must be a non-empty string.")

        with self._lock:
            if agent_id not in self._allocated_resources:
                self._allocated_resources[agent_id] = {res_type: 0 for res_type in self._total_resources}

    def request_resources(self, agent_id: str, requested_resources: Dict[str, Union[int, float]]) -> bool:
        """
        Attempts to allocate a set of resources to a specific agent.

        Args:
            agent_id (str): The unique identifier of the agent requesting resources.
            requested_resources (Dict[str, Union[int, float]]): A dictionary mapping
                                                                resource names to the
                                                                quantities requested.

        Returns:
            bool: True if all requested resources were successfully allocated, False otherwise.
        """
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("Agent ID must be a non-empty string.")
        self._validate_resource_request(requested_resources)

        with self._lock:
            if agent_id not in self._allocated_resources:
                # Optionally, auto-register agent or raise error
                # For now, let's auto-register for convenience but a strict system might require explicit registration
                self.register_agent(agent_id) # Uses the same lock, safe.

            # First, check if all requested resources are available
            for res_type, quantity in requested_resources.items():
                if res_type not in self._available_resources:
                    # Resource type not managed by this manager
                    return False
                if self._available_resources[res_type] < quantity:
                    # Not enough of this resource available
                    return False

            # If all checks pass, allocate the resources
            for res_type, quantity in requested_resources.items():
                self._available_resources[res_type] -= quantity
                self._allocated_resources[agent_id][res_type] = (
                    self._allocated_resources[agent_id].get(res_type, 0) + quantity
                )
            return True

    def release_resources(self, agent_id: str, released_resources: Dict[str, Union[int, float]]) -> bool:
        """
        Releases a set of resources previously allocated to an agent.

        Args:
            agent_id (str): The unique identifier of the agent releasing resources.
            released_resources (Dict[str, Union[int, float]]): A dictionary mapping
                                                                resource names to the
                                                                quantities being released.

        Returns:
            bool: True if all resources were successfully released, False otherwise.
        """
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("Agent ID must be a non-empty string.")
        self._validate_resource_request(released_resources)

        with self._lock:
            if agent_id not in self._allocated_resources:
                # Agent not registered or no resources allocated
                return False

            # Check if the agent actually holds these resources
            for res_type, quantity in released_resources.items():
                if res_type not in self._allocated_resources[agent_id]:
                    # Agent doesn't have this resource type allocated
                    return False
                if self._allocated_resources[agent_id][res_type] < quantity:
                    # Agent is trying to release more than it holds
                    return False

            # Release the resources
            for res_type, quantity in released_resources.items():
                self._allocated_resources[agent_id][res_type] -= quantity
                self._available_resources[res_type] += quantity
            return True

    def get_agent_allocations(self, agent_id: str) -> Dict[str, Union[int, float]]:
        """
        Gets the current resource allocations for a specific agent.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            Dict[str, Union[int, float]]: A dictionary of resources currently allocated
                                         to the agent, or an empty dictionary if the
                                         agent is not registered or has no allocations.
        """
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("Agent ID must be a non-empty string.")

        with self._lock:
            return self._allocated_resources.get(agent_id, {}).copy()

    def get_available_resources(self) -> Dict[str, Union[int, float]]:
        """
        Gets the currently available resources in the system.

        Returns:
            Dict[str, Union[int, float]]: A dictionary mapping resource names to their
                                         currently available quantities.
        """
        with self._lock:
            return self._available_resources.copy()

    def get_total_resources(self) -> Dict[str, Union[int, float]]:
        """
        Gets the total initial resources configured for the system.

        Returns:
            Dict[str, Union[int, float]]: A dictionary mapping resource names to their
                                         total quantities.
        """
        with self._lock:
            return self._total_resources.copy()

    def get_all_agent_allocations(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Gets the current resource allocations for all registered agents.

        Returns:
            Dict[str, Dict[str, Union[int, float]]]: A dictionary where keys are agent IDs
                                                     and values are their respective allocation dictionaries.
        """
        with self._lock:
            return {
                agent_id: allocations.copy()
                for agent_id, allocations in self._allocated_resources.items()
            }

    def reset_agent_allocations(self, agent_id: str) -> bool:
        """
        Resets all allocations for a specific agent, returning them to the available pool.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            bool: True if allocations were reset, False if agent was not found.
        """
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("Agent ID must be a non-empty string.")

        with self._lock:
            if agent_id not in self._allocated_resources:
                return False

            agent_current_allocations = self._allocated_resources.pop(agent_id)
            for res_type, quantity in agent_current_allocations.items():
                if res_type in self._available_resources:
                    self._available_resources[res_type] += quantity
            return True

    def __repr__(self):
        with self._lock:
            return (
                f"ResourceManager(Total: {self._total_resources}, "
                f"Available: {self._available_resources}, "
                f"Allocated: {self._allocated_resources})"
            )

```