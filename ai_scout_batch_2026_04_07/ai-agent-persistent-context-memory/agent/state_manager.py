```python
import logging
from typing import Dict, Any, Optional

# Assume these imports exist and are correctly structured based on context
# storage.database provides SessionLocal for database interaction
# storage.models provides the AgentState SQLAlchemy model
from storage.database import SessionLocal
from storage.models import AgentState as AgentStateModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages the persistent and in-memory state of a single AI agent.

    This class implements a strict singleton pattern based on `agent_id`,
    ensuring that only one StateManager instance exists for a given agent_id
    within the application's runtime. It handles loading, saving, and updating
    the agent's state from/to the persistent database.
    """
    _instance: Optional['StateManager'] = None
    _state: Dict[str, Any] = {}
    _agent_id: Optional[str] = None

    def __new__(cls, agent_id: str):
        """
        Ensures a single instance of StateManager per agent_id.

        Args:
            agent_id (str): A unique identifier for the AI agent whose state is being managed.
                            Must be a non-empty string.
        Raises:
            ValueError: If agent_id is not a non-empty string, or if an attempt
                        is made to initialize with a different agent_id than the
                        existing singleton instance.
        """
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string.")

        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
            cls._instance._agent_id = agent_id
            cls._instance._load_state_from_db()
        elif cls._instance._agent_id != agent_id:
            raise ValueError(
                f"StateManager is already initialized for agent '{cls._instance._agent_id}'. "
                f"Cannot re-initialize or retrieve for a different agent '{agent_id}' "
                "with the current singleton design. Consider redesigning for multi-agent support if needed."
            )
        return cls._instance

    def _load_state_from_db(self):
        """
        Loads the agent's state from the database. If no state is found,
        a new default state is initialized and saved.
        """
        db = SessionLocal()
        try:
            state_record = db.query(AgentStateModel).filter_by(agent_id=self._agent_id).first()
            if state_record:
                # Convert SQLAlchemy model to a plain dictionary for in-memory use
                self._state = {
                    "id": state_record.id,
                    "agent_id": state_record.agent_id,
                    "current_status": state_record.current_status,
                    "task_goals": state_record.task_goals,
                    "user_preferences": state_record.user_preferences,
                    "agent_configurations": state_record.agent_configurations,
                    "last_checkpoint_id": state_record.last_checkpoint_id,
                    "current_context_id": state_record.current_context_id,
                    "updated_at": state_record.updated_at,
                    "created_at": state_record.created_at,
                }
                logger.info(f"Loaded state for agent '{self._agent_id}'. State ID: {state_record.id}")
            else:
                self._state = {
                    "agent_id": self._agent_id,
                    "current_status": "initialized",
                    "task_goals": {},
                    "user_preferences": {},
                    "agent_configurations": {},
                    "last_checkpoint_id": None,
                    "current_context_id": None,
                }
                logger.info(f"No existing state found for agent '{self._agent_id}'. Initializing new state.")
                self._save_state_to_db() # Persist the initial state immediately
        except Exception as e:
            logger.exception(f"Error loading agent state for '{self._agent_id}'. Initializing default state.")
            # Fallback to a default state in case of any database error
            self._state = {
                "agent_id": self._agent_id,
                "current_status": "error_loading",
                "task_goals": {},
                "user_preferences": {},
                "agent_configurations": {},
                "last_checkpoint_id": None,
                "current_context_id": None,
            }
        finally:
            db.close()

    def _save_state_to_db(self):
        """
        Saves the current in-memory state to the database.
        Creates a new record if one doesn't exist for the agent_id, otherwise updates the existing one.
        """
        if self._agent_id is None:
            logger.error("Attempted to save state without an agent_id. StateManager not properly initialized.")
            return

        db = SessionLocal()
        try:
            state_record = db.query(AgentStateModel).filter_by(agent_id=self._agent_id).first()
            if state_record:
                # Update existing record
                state_record.current_status = self._state.get("current_status")
                state_record.task_goals = self._state.get("task_goals")
                state_record.user_preferences = self._state.get("user_preferences")
                state_record.agent_configurations = self._state.get("agent_configurations")
                state_record.last_checkpoint_id = self._state.get("last_checkpoint_id")
                state_record.current_context_id = self._state.get("current_context_id")
            else:
                # Create new record
                state_record = AgentStateModel(
                    agent_id=self._agent_id,
                    current_status=self._state.get("current_status", "initialized"),
                    task_goals=self._state.get("task_goals", {}),
                    user_preferences=self._state.get("user_preferences", {}),
                    agent_configurations=self._state.get("agent_configurations", {}),
                    last_checkpoint_id=self._state.get("last_checkpoint_id"),
                    current_context_id=self._state.get("current_context_id"),
                )
                db.add(state_record)
            db.commit()
            # Refresh the record to get any ORM-managed updates (like `id` for new records, `updated_at`)
            db.refresh(state_record)
            self._state["id"] = state_record.id
            self._state["updated_at"] = state_record.updated_at
            self._state["created_at"] = state_record.created_at # Ensure created_at is also reflected if new
            logger.debug(f"State for agent '{self._agent_id}' saved to database. State ID: {state_record.id}")
        except Exception as e:
            db.rollback()
            logger.exception(f"Error saving agent state for '{self._agent_id}'. Rolling back transaction.")
        finally:
            db.close()

    def get_state(self, key: Optional[str] = None) -> Any:
        """
        Retrieves the entire state dictionary or a specific key's value from the current in-memory state.
        Returns a copy of mutable objects (dicts, lists) to prevent external direct modification
        of the internal state.

        Args:
            key (Optional[str]): The specific key to retrieve. If None, the entire state is returned.

        Returns:
            Any: The value associated with the key, or the full state dictionary.
        """
        if key:
            value = self._state.get(key)
            return value.copy() if isinstance(value, (dict, list)) else value
        return self._state.copy()

    def update_state(self, updates: Dict[str, Any], persist: bool = True):
        """
        Updates the agent's state with new values from the provided dictionary.
        Only fields explicitly in 'updates' will be modified.
        Optionally persists the changes to the database immediately if `persist` is True.

        Args:
            updates (Dict[str, Any]): A dictionary of key-value pairs to update in the state.
            persist (bool): If True, immediately saves the updated state to the database.
                            Defaults to True.
        """
        if not updates:
            logger.debug("No updates provided, skipping state update.")
            return

        has_changed = False
        for key, value in updates.items():
            # Perform a shallow comparison for change detection.
            # For nested mutable types, a deep comparison might be desired in production.
            if self._state.get(key) != value:
                self._state[key] = value
                has_changed = True

        if persist and has_changed:
            self._save_state_to_db()
        elif persist and not has_changed:
            logger.debug(f"State for agent '{self._agent_id}' has not changed for the given keys, skipping database save.")

    def set_agent_status(self, status: str, persist: bool = True):
        """
        Helper method to update the agent's current operational status.

        Args:
            status (str): The new status string (e.g., 'idle', 'executing', 'awaiting_feedback').
            persist (bool): If True, immediately saves the state to the database.
        Raises:
            ValueError: If status is not a non-empty string.
        """
        if not isinstance(status, str) or not status:
            raise ValueError("Status must be a non-empty string.")
        self.update_state({"current_status": status}, persist=persist)

    def set_task_goals(self, goals: Dict[str, Any], persist: bool = True):
        """
        Helper method to update the agent's current task goals.

        Args:
            goals (Dict[str, Any]): A dictionary representing the agent's current task goals.
            persist (bool): If True, immediately saves the state to the database.
        Raises:
            ValueError: If goals is not a dictionary.
        """
        if not isinstance(goals, dict):
            raise ValueError("Goals must be a dictionary.")
        self.update_state({"task_goals": goals}, persist=persist)

    def set_user_preferences(self, preferences: Dict[str, Any], persist: bool = True):
        """
        Helper method to update the agent's user preferences.

        Args:
            preferences (Dict[str, Any]): A dictionary representing user preferences.
            persist (bool): If True, immediately saves the state to the database.
        Raises:
            ValueError: If preferences is not a dictionary.
        """
        if not isinstance(preferences, dict):
            raise ValueError("Preferences must be a dictionary.")
        self.update_state({"user_preferences": preferences}, persist=persist)

    def set_agent_configurations(self, configs: Dict[str, Any], persist: bool = True):
        """
        Helper method to update the agent's operational configurations.

        Args:
            configs (Dict[str, Any]): A dictionary representing agent configurations.
            persist (bool): If True, immediately saves the state to the database.
        Raises:
            ValueError: If configs is not a dictionary.
        """
        if not isinstance(configs, dict):
            raise ValueError("Configurations must be a dictionary.")
        self.update_state({"agent_configurations": configs}, persist=persist)

    def set_last_checkpoint_id(self, checkpoint_id: Optional[int], persist: bool = True):
        """
        Helper method to record the ID of the last confirmed checkpoint.

        Args:
            checkpoint_id (Optional[int]): The integer ID of the checkpoint, or None if cleared.
            persist (bool): If True, immediately saves the state to the database.
        Raises:
            ValueError: If checkpoint_id is not an integer or None.
        """
        if not (isinstance(checkpoint_id, int) or checkpoint_id is None):
            raise ValueError("Checkpoint ID must be an integer or None.")
        self.update_state({"last_checkpoint_id": checkpoint_id}, persist=persist)

    def set_current_context_id(self, context_id: Optional[str], persist: bool = True):
        """
        Helper method to record the ID/reference of the current active context (e.g., a tree node ID).

        Args:
            context_id (Optional[str]): The string ID/reference of the current context, or None.
            persist (bool): If True, immediately saves the state to the database.
        Raises:
            ValueError: If context_id is not a string or None.
        """
        if not (isinstance(context_id, str) or context_id is None):
            raise ValueError("Context ID must be a string or None.")
        self.update_state({"current_context_id": context_id}, persist=persist)

    def __repr__(self):
        """Provides a developer-friendly representation of the StateManager."""
        return f"StateManager(agent_id='{self._agent_id}', status='{self._state.get('current_status')}')"

    def __str__(self):
        """Provides a human-readable string representation of the agent's key state."""
        # Omit internal IDs and timestamps for cleaner string representation
        display_state = {k: v for k, v in self._state.items() if k not in ["id", "created_at", "updated_at"]}
        return f"Agent State for '{self._agent_id}': {display_state}"

```