```python
import threading
import copy
from typing import Dict, Any, Optional, List

class StateManager:
    """
    Manages the canonical state of all agents and the overall system.
    Provides thread-safe access and modification of both global and agent-specific states.
    Facilitates predictability and potential for rollback by centralizing state management.
    """

    def __init__(self):
        """
        Initializes the StateManager with empty global and agent states.
        A reentrant lock (RLock) is used to ensure thread-safe access to the state,
        allowing the same thread to acquire the lock multiple times if needed.
        """
        self._global_state: Dict[str, Any] = {}
        self._agent_states: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        # self._state_history = [] # Placeholder for future rollback capability

    def get_global_state(self, key: Optional[str] = None) -> Any:
        """
        Retrieves the entire global state or the value of a specific key from it.
        Returns a deep copy to prevent external modification of the internal state.

        Args:
            key (Optional[str]): The key to retrieve. If None, returns the entire global state.

        Returns:
            Any: The global state dictionary, or the value associated with the key.
                 Returns None if the key does not exist.
        """
        with self._lock:
            if key is None:
                return copy.deepcopy(self._global_state)
            return copy.deepcopy(self._global_state.get(key))

    def update_global_state(self, key: str, value: Any) -> None:
        """
        Updates a specific key-value pair in the global state.
        If the key does not exist, it will be added.

        Args:
            key (str): The key to update. Must be a string.
            value (Any): The new value for the key.

        Raises:
            TypeError: If the key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError(f"Global state key must be a string, got {type(key).__name__}.")

        with self._lock:
            # Future: Consider logging old_value for debugging/history
            # old_value = self._global_state.get(key)
            self._global_state[key] = value
            # self._record_state_change("global", None, key, old_value, value)

    def set_global_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Replaces the entire global state with a new dictionary.
        The new dictionary is deep-copied to prevent direct modification by external references.

        Args:
            state_dict (Dict[str, Any]): The new dictionary for the global state.

        Raises:
            TypeError: If the provided state_dict is not a dictionary.
        """
        if not isinstance(state_dict, dict):
            raise TypeError(f"Global state must be a dictionary, got {type(state_dict).__name__}.")

        with self._lock:
            self._global_state = copy.deepcopy(state_dict)

    def get_agent_state(self, agent_id: str, key: Optional[str] = None) -> Any:
        """
        Retrieves the entire state for a specific agent or the value of a specific key from it.
        Returns a deep copy to prevent external modification of the internal state.

        Args:
            agent_id (str): The unique identifier of the agent. Must be a string.
            key (Optional[str]): The key to retrieve. If None, returns the entire agent's state.

        Returns:
            Any: The agent's state dictionary, or the value associated with the key.
                 Returns None if the agent or the specified key does not exist.

        Raises:
            TypeError: If the agent_id is not a string.
        """
        if not isinstance(agent_id, str):
            raise TypeError(f"Agent ID must be a string, got {type(agent_id).__name__}.")

        with self._lock:
            agent_state = self._agent_states.get(agent_id)
            if agent_state is None:
                return None  # Agent does not exist

            if key is None:
                return copy.deepcopy(agent_state)
            return copy.deepcopy(agent_state.get(key))

    def update_agent_state(self, agent_id: str, key: str, value: Any) -> None:
        """
        Updates a specific key-value pair in an agent's state.
        If the agent's state does not exist, it will be initialized as an empty dictionary.
        If the key does not exist for the agent, it will be added.

        Args:
            agent_id (str): The unique identifier of the agent. Must be a string.
            key (str): The key to update. Must be a string.
            value (Any): The new value for the key.

        Raises:
            TypeError: If agent_id or key is not a string.
        """
        if not isinstance(agent_id, str):
            raise TypeError(f"Agent ID must be a string, got {type(agent_id).__name__}.")
        if not isinstance(key, str):
            raise TypeError(f"Agent state key must be a string, got {type(key).__name__}.")

        with self._lock:
            if agent_id not in self._agent_states:
                self._agent_states[agent_id] = {}
            # Future: Consider logging old_value for debugging/history
            # old_value = self._agent_states[agent_id].get(key)
            self._agent_states[agent_id][key] = value
            # self._record_state_change("agent", agent_id, key, old_value, value)

    def set_agent_state(self, agent_id: str, state_dict: Dict[str, Any]) -> None:
        """
        Replaces the entire state for a specific agent with a new dictionary.
        If the agent's state does not exist, it will be created.
        The new dictionary is deep-copied to prevent direct modification by external references.

        Args:
            agent_id (str): The unique identifier of the agent. Must be a string.
            state_dict (Dict[str, Any]): The new dictionary for the agent's state.

        Raises:
            TypeError: If agent_id is not a string or state_dict is not a dictionary.
        """
        if not isinstance(agent_id, str):
            raise TypeError(f"Agent ID must be a string, got {type(agent_id).__name__}.")
        if not isinstance(state_dict, dict):
            raise TypeError(f"Agent state must be a dictionary, got {type(state_dict).__name__}.")

        with self._lock:
            self._agent_states[agent_id] = copy.deepcopy(state_dict)

    def remove_agent_state(self, agent_id: str) -> bool:
        """
        Removes the state of a specific agent from the manager.

        Args:
            agent_id (str): The unique identifier of the agent. Must be a string.

        Returns:
            bool: True if the agent's state was successfully removed, False if it didn't exist.

        Raises:
            TypeError: If the agent_id is not a string.
        """
        if not isinstance(agent_id, str):
            raise TypeError(f"Agent ID must be a string, got {type(agent_id).__name__}.")

        with self._lock:
            if agent_id in self._agent_states:
                del self._agent_states[agent_id]
                return True
            return False

    def get_all_agent_ids(self) -> List[str]:
        """
        Returns a list of all agent IDs currently tracked by the state manager.
        The list is a copy to prevent external modification.

        Returns:
            List[str]: A list of agent IDs.
        """
        with self._lock:
            return list(self._agent_states.keys())

    # --- Future Enhancements (commented out for basic prototype) ---
    # def _record_state_change(self, entity_type: str, entity_id: Optional[str], key: str, old_value: Any, new_value: Any):
    #     """
    #     Internal method to record state changes for history/rollback or logging.
    #     This would typically push to a history list or emit an event on an event bus.
    #     """
    #     change_record = {
    #         "timestamp": time.time(),
    #         "entity_type": entity_type,
    #         "entity_id": entity_id,
    #         "key": key,
    #         "old_value": old_value,
    #         "new_value": new_value
    #     }
    #     self._state_history.append(change_record)
    #     # Example: Emit an event
    #     # from core.event_bus import EventBus
    #     # EventBus.get_instance().emit("state_changed", change_record)

    # def rollback_global_state(self, timestamp: float) -> None:
    #     """
    #     Rolls back the global state to a previous point in time based on history.
    #     (Requires _state_history to be implemented and properly managed)
    #     """
    #     raise NotImplementedError("Rollback functionality is not yet implemented.")

    # def rollback_agent_state(self, agent_id: str, timestamp: float) -> None:
    #     """
    #     Rolls back an agent's state to a previous point in time based on history.
    #     (Requires _state_history to be implemented and properly managed)
    #     """
    #     raise NotImplementedError("Rollback functionality is not yet implemented.")

```