```python
import json
import os
import uuid
import hashlib
from typing import Dict, Any, Optional

class StateManager:
    """
    Manages and serializes internal states of agentic systems.
    These states can be explicitly referenced or re-injected in subsequent LLM calls
    to enhance determinism and contextual anchoring.

    The manager handles persistence of states to a file, allowing for consistent
    retrieval and re-use across different runs or agent steps.
    """

    def __init__(self, storage_path: str):
        """
        Initializes the StateManager with a path to store state data.

        Args:
            storage_path (str): The file path (e.g., 'data/agent_states.json')
                                where states will be loaded from and saved to.
                                The directory for this path will be created if it doesn't exist.
        """
        self.storage_path = storage_path
        self._states: Dict[str, Dict[str, Any]] = {}
        self._load_states()

    def _load_states(self) -> None:
        """
        Loads states from the configured storage path.
        Handles cases where the file does not exist, is empty, or is malformed.
        """
        if not os.path.exists(self.storage_path):
            # Ensure the directory exists if the file doesn't
            os.makedirs(os.path.dirname(self.storage_path) or '.', exist_ok=True)
            self._states = {}
            return

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:  # Handle empty file
                    self._states = {}
                else:
                    self._states = json.loads(content)
            # Basic validation: ensure loaded content is a dictionary
            if not isinstance(self._states, dict):
                print(f"Warning: Content of {self.storage_path} is not a dictionary. "
                      f"Starting with empty states.")
                self._states = {}
        except json.JSONDecodeError as e:
            print(f"Warning: Could not decode JSON from {self.storage_path}. "
                  f"Starting with empty states. Error: {e}")
            self._states = {}
        except IOError as e:
            print(f"Warning: Could not read file {self.storage_path}. "
                  f"Starting with empty states. Error: {e}")
            self._states = {}

    def _save_states(self) -> None:
        """
        Saves the current states to the configured storage path.
        """
        try:
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(self.storage_path) or '.', exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self._states, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Error: Could not save states to {self.storage_path}. Error: {e}")
        except TypeError as e:
            print(f"Error: State data contains non-serializable objects. Cannot save states. Error: {e}")


    def add_state(self, state_data: Dict[str, Any], state_id: Optional[str] = None) -> str:
        """
        Adds a new state to the manager. If no state_id is provided, a unique one is generated.

        Args:
            state_data (Dict[str, Any]): The actual state data to store. This data should be
                                         JSON-serializable to ensure persistence.
            state_id (Optional[str]): An optional unique identifier for the state.
                                      If None, a UUID will be generated.

        Returns:
            str: The ID of the added state.

        Raises:
            ValueError: If a state with the given state_id already exists.
            TypeError: If the state_data is not JSON-serializable.
        """
        if state_id is None:
            state_id = uuid.uuid4().hex
        elif state_id in self._states:
            raise ValueError(f"State with ID '{state_id}' already exists. Use update_state to modify.")

        # Pre-check for serializability before adding
        try:
            json.dumps(state_data)
        except TypeError as e:
            raise TypeError(f"Provided state_data is not JSON-serializable: {e}") from e

        self._states[state_id] = state_data
        self._save_states()
        return state_id

    def get_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a state by its ID.

        Args:
            state_id (str): The ID of the state to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The state data if found, otherwise None.
        """
        return self._states.get(state_id)

    def update_state(self, state_id: str, new_state_data: Dict[str, Any]) -> bool:
        """
        Updates an existing state.

        Args:
            state_id (str): The ID of the state to update.
            new_state_data (Dict[str, Any]): The new state data to replace the old one.
                                             This data should be JSON-serializable.

        Returns:
            bool: True if the state was updated, False if the state_id was not found.

        Raises:
            TypeError: If the new_state_data is not JSON-serializable.
        """
        if state_id in self._states:
            # Pre-check for serializability before updating
            try:
                json.dumps(new_state_data)
            except TypeError as e:
                raise TypeError(f"Provided new_state_data is not JSON-serializable: {e}") from e

            self._states[state_id] = new_state_data
            self._save_states()
            return True
        return False

    def delete_state(self, state_id: str) -> bool:
        """
        Deletes a state by its ID.

        Args:
            state_id (str): The ID of the state to delete.

        Returns:
            bool: True if the state was deleted, False if the state_id was not found.
        """
        if state_id in self._states:
            del self._states[state_id]
            self._save_states()
            return True
        return False

    def serialize_state(self, state_id: str, compact: bool = True) -> Optional[str]:
        """
        Serializes a specific state into a JSON string, suitable for injection into a prompt.

        Args:
            state_id (str): The ID of the state to serialize.
            compact (bool): If True, returns a compact JSON string (no indentation, minimal separators).
                            If False, returns a pretty-printed JSON string (indent=4).
                            Compact is generally better for prompt injection.

        Returns:
            Optional[str]: The JSON string representation of the state, or None if not found.
        """
        state_data = self.get_state(state_id)
        if state_data:
            try:
                if compact:
                    # Compact format: no indentation, minimal separators
                    return json.dumps(state_data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
                else:
                    # Pretty-printed format
                    return json.dumps(state_data, sort_keys=True, indent=4, ensure_ascii=False)
            except TypeError as e:
                print(f"Error: Could not serialize state '{state_id}'. State data might not be JSON serializable. Error: {e}")
                return None
        return None

    def get_state_hash(self, state_id: str) -> Optional[str]:
        """
        Generates a cryptographic hash (SHA256) of a state's serialized content.
        This hash provides an immutable reference and can be used for checking state integrity
        or for simple versioning within prompts.

        Args:
            state_id (str): The ID of the state for which to generate the hash.

        Returns:
            Optional[str]: The SHA256 hash string of the state, or None if the state is not found
                           or cannot be serialized. The hash is generated from a compact,
                           sorted-key JSON representation to ensure consistency.
        """
        # Ensure consistent serialization for hashing
        serialized_state = self.serialize_state(state_id, compact=True)
        if serialized_state:
            return hashlib.sha256(serialized_state.encode('utf-8')).hexdigest()
        return None
```