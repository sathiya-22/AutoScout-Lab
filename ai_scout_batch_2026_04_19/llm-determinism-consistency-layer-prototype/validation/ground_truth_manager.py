```python
import json
import os
from typing import Any, Dict, List, Optional

class GroundTruthManager:
    """
    Manages a store of expected outputs or ground truth data for specific prompts/scenarios.
    This data is used for validation and correction of LLM outputs.

    Ground truth data is stored in a JSON file and loaded/saved automatically.
    """

    def __init__(self, ground_truth_file_path: str):
        """
        Initializes the GroundTruthManager.
        Loads ground truth data from the specified JSON file. If the file does not exist
        or is corrupted, an empty store is initialized.

        Args:
            ground_truth_file_path (str): The path to the JSON file where ground truth data is stored.
                                          This path should typically reside within the `data/` directory.
        """
        self.ground_truth_file_path = ground_truth_file_path
        self._ground_truth_store: Dict[str, Any] = {}
        self._load_from_file()

    def _load_from_file(self) -> None:
        """
        Loads ground truth data from the JSON file into the in-memory store.
        Handles cases where the file does not exist, is empty, or is corrupted,
        initializing an empty store in such scenarios.
        """
        if not os.path.exists(self.ground_truth_file_path):
            print(f"Ground truth file not found at '{self.ground_truth_file_path}'. Initializing with empty store.")
            return

        try:
            with open(self.ground_truth_file_path, 'r', encoding='utf-8') as f:
                # Check if the file is empty before attempting to load JSON
                file_content = f.read()
                if not file_content.strip():
                    print(f"Ground truth file '{self.ground_truth_file_path}' is empty. Initializing with empty store.")
                    self._ground_truth_store = {}
                    return
                self._ground_truth_store = json.loads(file_content)
            print(f"Successfully loaded ground truth from '{self.ground_truth_file_path}'.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from '{self.ground_truth_file_path}': {e}. Initializing with empty store.")
            self._ground_truth_store = {}
        except IOError as e:
            print(f"IOError occurred while loading ground truth from '{self.ground_truth_file_path}': {e}. Initializing with empty store.")
            self._ground_truth_store = {}
        except Exception as e:
            print(f"An unexpected error occurred while loading ground truth from '{self.ground_truth_file_path}': {e}. Initializing with empty store.")
            self._ground_truth_store = {}

    def _save_to_file(self) -> None:
        """
        Saves the current in-memory ground truth store to the JSON file.
        Ensures the directory path exists before attempting to write the file.
        """
        # Ensure the directory exists
        dir_path = os.path.dirname(self.ground_truth_file_path)
        if dir_path:  # Only create if dir_path is not empty (i.e., not just a filename)
            os.makedirs(dir_path, exist_ok=True)

        try:
            with open(self.ground_truth_file_path, 'w', encoding='utf-8') as f:
                json.dump(self._ground_truth_store, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved ground truth to '{self.ground_truth_file_path}'.")
        except IOError as e:
            print(f"Error saving ground truth to '{self.ground_truth_file_path}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred while saving ground truth to '{self.ground_truth_file_path}': {e}")

    def get_ground_truth(self, scenario_id: str) -> Optional[Any]:
        """
        Retrieves the ground truth data for a given scenario ID.

        Args:
            scenario_id (str): The unique identifier for the scenario.

        Returns:
            Optional[Any]: The ground truth data associated with the scenario_id,
                           or None if the scenario_id is not found in the store.
        """
        if scenario_id not in self._ground_truth_store:
            # print(f"Warning: Scenario ID '{scenario_id}' not found in ground truth store.")
            return None
        return self._ground_truth_store.get(scenario_id)

    def add_ground_truth(self, scenario_id: str, truth_data: Any) -> None:
        """
        Adds or updates ground truth data for a specific scenario.
        The changes are immediately saved to the persistent file.

        Args:
            scenario_id (str): The unique identifier for the scenario.
            truth_data (Any): The ground truth data (can be a string, dict, list,
                              or any JSON-serializable type).
        """
        if not isinstance(scenario_id, str) or not scenario_id:
            print(f"Error: Invalid scenario_id '{scenario_id}'. Must be a non-empty string.")
            return

        self._ground_truth_store[scenario_id] = truth_data
        self._save_to_file()
        print(f"Ground truth for scenario '{scenario_id}' added/updated.")

    def remove_ground_truth(self, scenario_id: str) -> bool:
        """
        Removes ground truth data for a specific scenario.
        The changes are immediately saved to the persistent file.

        Args:
            scenario_id (str): The unique identifier for the scenario to remove.

        Returns:
            bool: True if the scenario was found and removed, False otherwise.
        """
        if not isinstance(scenario_id, str) or not scenario_id:
            print(f"Error: Invalid scenario_id '{scenario_id}'. Must be a non-empty string.")
            return False

        if scenario_id in self._ground_truth_store:
            del self._ground_truth_store[scenario_id]
            self._save_to_file()
            print(f"Ground truth for scenario '{scenario_id}' removed.")
            return True
        print(f"Warning: Attempted to remove non-existent scenario ID '{scenario_id}'.")
        return False

    def list_scenarios(self) -> List[str]:
        """
        Returns a list of all scenario IDs currently managed by the store.

        Returns:
            List[str]: A list of scenario IDs.
        """
        return list(self._ground_truth_store.keys())

    def get_all_ground_truth(self) -> Dict[str, Any]:
        """
        Returns a copy of all ground truth data currently managed by the store.
        This can be useful for comprehensive review or debugging.

        Returns:
            Dict[str, Any]: A dictionary containing all ground truth data,
                            keyed by scenario ID.
        """
        return self._ground_truth_store.copy()

```