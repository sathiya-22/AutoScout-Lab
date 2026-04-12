import datetime
from collections import deque
from typing import Any, Dict, List, Optional, Deque

class StateManager:
    """
    Manages the historical trace of an AI agent's internal states and external environment feedback.
    Acts as the monitor's memory, aggregating observations for analysis by detectors.
    """
    def __init__(self, max_history_length: Optional[int] = None):
        """
        Initializes the StateManager.

        Args:
            max_history_length (Optional[int]): The maximum number of observations
                                                to retain in history for each agent.
                                                If None, history length is unlimited.
        """
        self._history: Dict[str, Deque[Dict[str, Any]]] = {}
        self._max_history_length = max_history_length

    def add_observation(self, agent_id: str, step_type: str, data: Any, timestamp: Optional[datetime.datetime] = None):
        """
        Adds a new observation to the historical trace for a specific agent.

        Args:
            agent_id (str): A unique identifier for the agent being monitored.
            step_type (str): The type of observation (e.g., 'thought', 'tool_call', 'tool_result', 'output').
            data (Any): The actual data associated with the observation (e.g., thought string,
                        dictionary of tool call arguments, tool execution result).
            timestamp (Optional[datetime.datetime]): The time the observation occurred.
                                                    If None, the current UTC time is used.
        """
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string.")
        if not isinstance(step_type, str) or not step_type:
            raise ValueError("step_type must be a non-empty string.")

        if agent_id not in self._history:
            self._history[agent_id] = deque(maxlen=self._max_history_length)

        observation = {
            "timestamp": timestamp if timestamp is not None else datetime.datetime.now(datetime.timezone.utc),
            "step_type": step_type,
            "data": data,
            "agent_id": agent_id,
        }
        self._history[agent_id].append(observation)

    def get_full_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the complete historical trace for a given agent.

        Args:
            agent_id (str): The identifier of the agent whose history is requested.

        Returns:
            List[Dict[str, Any]]: A list of observation dictionaries, ordered chronologically.
                                  Returns an empty list if the agent_id is not found.
        """
        if not isinstance(agent_id, str) or not agent_id:
            return [] # Or raise ValueError, depending on desired strictness

        return list(self._history.get(agent_id, []))

    def get_last_n_observations(self, agent_id: str, n: int) -> List[Dict[str, Any]]:
        """
        Retrieves the 'n' most recent observations for a given agent.

        Args:
            agent_id (str): The identifier of the agent whose history is requested.
            n (int): The number of recent observations to retrieve. Must be a non-negative integer.

        Returns:
            List[Dict[str, Any]]: A list of the 'n' most recent observation dictionaries,
                                  ordered chronologically from oldest to newest within the 'n' window.
                                  Returns fewer than 'n' observations if the total history is smaller.
                                  Returns an empty list if n <= 0 or the agent_id is not found.
        """
        if not isinstance(agent_id, str) or not agent_id:
            return []
        if not isinstance(n, int) or n < 0:
            return []

        agent_history = self._history.get(agent_id, deque())
        if n == 0:
            return []
        # Deque slices inefficiently, converting to list first.
        # If n is larger than the actual history, it will return the full history.
        return list(agent_history)[-n:]

    def clear_history(self, agent_id: Optional[str] = None):
        """
        Clears the historical trace for a specific agent, or for all agents if no agent_id is provided.

        Args:
            agent_id (Optional[str]): The identifier of the agent whose history to clear.
                                    If None, all agents' histories are cleared.
        """
        if agent_id is None:
            self._history.clear()
        elif agent_id in self._history:
            self._history[agent_id].clear()
        # If agent_id is not found, do nothing, which is fine.