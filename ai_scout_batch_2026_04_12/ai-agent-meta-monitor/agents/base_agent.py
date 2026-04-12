```python
import abc
from typing import Any, Dict, List, Optional

class BaseAgent(abc.ABC):
    """
    Abstract Base Class for AI agents.

    This interface defines the core capabilities expected from any agent that
    will be monitored by the meta-monitoring framework. It includes methods
    for executing tasks, exposing internal state, and responding to interventions.
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the BaseAgent.

        Args:
            agent_id (str): A unique identifier for the agent instance.
            config (Optional[Dict[str, Any]]): Configuration parameters for the agent.
        
        Raises:
            ValueError: If agent_id is not a non-empty string.
        """
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string.")
        
        self.agent_id: str = agent_id
        self.config: Dict[str, Any] = config if config is not None else {}
        self._current_task: Optional[str] = None
        # _internal_execution_log is for the agent's own internal record-keeping,
        # separate from what the monitor collects via instrumentation.
        self._internal_execution_log: List[Dict[str, Any]] = []

    def set_task(self, task_description: str) -> None:
        """
        Sets the current task for the agent.

        Args:
            task_description (str): A textual description of the task to be performed.
        
        Raises:
            ValueError: If task_description is not a non-empty string.
        """
        if not isinstance(task_description, str) or not task_description.strip():
            raise ValueError("task_description must be a non-empty string.")
        
        self._current_task = task_description
        self._internal_execution_log = []  # Reset internal log for a new task

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Executes the agent's main task.

        This method should encapsulate the agent's full execution loop.
        It is expected that internal steps within this run method (e.g., tool calls,
        thought generation, output production) will be instrumented by the monitor
        framework for observation.

        Returns:
            Any: The final output or result of the agent's task.
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'run' method.")

    @abc.abstractmethod
    def step(self, *args, **kwargs) -> Any:
        """
        Executes a single logical step of the agent's operation.

        This method is designed to be called repeatedly by an external orchestrator
        (like the monitor core) to advance the agent's state. It allows the monitor
        to observe and intervene at a fine-grained level.
        The agent should make some discernible progress in a single step.

        Returns:
            Any: The result or state change from this single step. This could be
                 a thought, a tool call result, a partial output, or an indication
                 of completion/stalling. The specific return type depends on the
                 concrete agent implementation.
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'step' method.")

    @abc.abstractmethod
    def get_internal_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary representing the agent's current critical internal state.

        This state is observed by the monitor to detect issues.
        Examples of state include: 'current_thought', 'last_tool_call', 'last_tool_result',
        'current_output', 'current_task_status', 'plan_history_summary'.
        It should provide sufficient detail for a Detector to analyze progress and patterns.

        Returns:
            Dict[str, Any]: A dictionary containing key-value pairs of internal state.
                            Must return a dictionary, even if empty.
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'get_internal_state' method.")

    @abc.abstractmethod
    def replan(self, new_context: Optional[str] = None) -> None:
        """
        Triggers the agent's re-planning mechanism.

        This method is called by an intervention to guide the agent out of a
        problematic state by forcing it to re-evaluate its plan. The agent should
        discard its current plan (or parts of it) and formulate a new one.

        Args:
            new_context (Optional[str]): Additional context or revised instructions
                                         to guide the re-planning process. If None,
                                         the agent should attempt to replan based
                                         on its current understanding and history.
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'replan' method.")

    @abc.abstractmethod
    def receive_hint(self, hint: str) -> None:
        """
        Provides a contextual hint or instruction to the agent.

        This method is called by an intervention to inject specific guidance
        into the agent's ongoing process, without necessarily forcing a full replan.
        The agent should integrate this hint into its subsequent reasoning or action selection.

        Args:
            hint (str): The contextual hint or instruction.
        
        Raises:
            ValueError: If hint is not a non-empty string.
            NotImplementedError: If the subclass does not implement this method.
        """
        if not isinstance(hint, str) or not hint.strip():
            raise ValueError("Hint must be a non-empty string.")
        raise NotImplementedError("Subclasses must implement the 'receive_hint' method.")

    @abc.abstractmethod
    def is_task_complete(self) -> bool:
        """
        Checks if the agent has completed its assigned task successfully.

        Returns:
            bool: True if the task is complete, False otherwise.
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'is_task_complete' method.")

    @property
    def current_task(self) -> Optional[str]:
        """
        Returns the current task description assigned to the agent.
        """
        return self._current_task

    def _log_action(self, action_type: str, details: Dict[str, Any]) -> None:
        """
        Internal method for the agent to log its own actions for debugging or
        internal reasoning purposes. This log is typically not directly consumed
        by the external monitor, which relies on its own instrumentation.

        Args:
            action_type (str): A descriptive string for the action (e.g., "thought", "tool_call").
            details (Dict[str, Any]): A dictionary containing specific details about the action.
        """
        import time
        self._internal_execution_log.append({
            "timestamp": time.time(),
            "action_type": action_type,
            "details": details
        })

    def __repr__(self) -> str:
        """
        Returns a string representation of the agent, primarily its class name and ID.
        """
        return f"{self.__class__.__name__}(id='{self.agent_id}')"

```