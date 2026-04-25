```python
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Awaitable

# Setup basic logging for the agent module.
# In a full system, this would integrate with a centralized logging solution.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    BaseAgent defines the core abstraction for an AI agent in the multi-agent system.
    Agents are designed to be stateful, capable of asynchronous operations, and equipped
    with a standard interface for communication and task handling.

    Each agent has a unique ID, a human-readable name, and maintains an internal context
    for its state. It provides abstract methods that must be implemented by concrete
    agent types to define their specific behaviors and processing logic.
    """

    def __init__(self, agent_id: Optional[str] = None, name: Optional[str] = None):
        """
        Initializes a new BaseAgent instance.

        Args:
            agent_id (str, optional): A unique identifier for the agent. If None, a UUID will be generated.
            name (str, optional): A human-readable name for the agent. If None, a default name is generated.
        """
        self._id: str = agent_id if agent_id else str(uuid.uuid4())
        self._name: str = name if name else f"Agent-{self._id[:8]}"
        self._context: Dict[str, Any] = {}  # Internal state/context for the agent
        self._is_running: bool = False

        # Callbacks or hooks for interacting with the orchestration layer.
        # These are typically set by the Orchestrator after agent creation
        # to allow agents to send messages or request tasks without direct
        # knowledge of the communication or task management implementation.
        self._message_sender: Optional[Callable[[str, Any], Awaitable[None]]] = None
        self._task_requester: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None

        logger.info(f"Agent '{self.name}' (ID: {self.id}) initialized.")

    @property
    def id(self) -> str:
        """Returns the unique identifier of the agent."""
        return self._id

    @property
    def name(self) -> str:
        """Returns the human-readable name of the agent."""
        return self._name

    @property
    def context(self) -> Dict[str, Any]:
        """
        Returns the internal context (state) of the agent.
        This dictionary can be used to store any agent-specific data or state variables.
        """
        return self._context

    @context.setter
    def context(self, value: Dict[str, Any]):
        """
        Sets the internal context (state) of the agent.
        Raises a TypeError if the provided value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("Agent context must be a dictionary.")
        self._context = value

    @property
    def is_running(self) -> bool:
        """Returns True if the agent is currently running, False otherwise."""
        return self._is_running

    def set_message_sender(self, sender: Callable[[str, Any], Awaitable[None]]):
        """
        Sets the asynchronous callback function used by the agent to send messages.
        This function is typically provided by the Orchestrator or a communication manager.
        The sender function should accept two arguments: `target_agent_id` (str) and `message` (Any).
        """
        if not callable(sender):
            raise TypeError("Message sender must be a callable.")
        self._message_sender = sender
        logger.debug(f"Agent '{self.name}' message sender configured.")

    def set_task_requester(self, requester: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Sets the asynchronous callback function used by the agent to request new tasks.
        This function is typically provided by the Orchestrator or a task manager.
        The requester function should accept one argument: `task_details` (Dict[str, Any]).
        """
        if not callable(requester):
            raise TypeError("Task requester must be a callable.")
        self._task_requester = requester
        logger.debug(f"Agent '{self.name}' task requester configured.")

    async def send_message(self, target_agent_id: str, message: Any):
        """
        Asynchronously sends a message to another agent identified by `target_agent_id`.
        This method delegates the actual sending to the configured `_message_sender` callback.

        Args:
            target_agent_id (str): The ID of the recipient agent.
            message (Any): The message content to send.
        """
        if self._message_sender:
            try:
                await self._message_sender(target_agent_id, message)
                logger.debug(f"Agent '{self.name}' (ID: {self.id}) sent message to '{target_agent_id}'.")
            except Exception as e:
                logger.error(
                    f"Agent '{self.name}' (ID: {self.id}) failed to send message to '{target_agent_id}': {e}",
                    exc_info=True
                )
        else:
            logger.warning(
                f"Agent '{self.name}' (ID: {self.id}) has no message sender configured. "
                "Message not sent to '{target_agent_id}'."
            )

    async def request_task(self, task_details: Dict[str, Any]):
        """
        Asynchronously requests a new task to be handled by the orchestration layer.
        This method delegates the task request to the configured `_task_requester` callback.

        Args:
            task_details (Dict[str, Any]): A dictionary containing details of the requested task.
        """
        if self._task_requester:
            try:
                await self._task_requester(task_details)
                logger.debug(f"Agent '{self.name}' (ID: {self.id}) requested a new task: {task_details.get('type')}")
            except Exception as e:
                logger.error(
                    f"Agent '{self.name}' (ID: {self.id}) failed to request task '{task_details.get('type')}': {e}",
                    exc_info=True
                )
        else:
            logger.warning(
                f"Agent '{self.name}' (ID: {self.id}) has no task requester configured. "
                "Task '{task_details.get('type')}' not requested."
            )

    @abstractmethod
    async def start(self):
        """
        Asynchronously starts the agent. This method must be implemented by subclasses
        to define agent-specific startup logic, such as initializing resources or
        starting internal loops.

        It should set `_is_running` to True.
        """
        if self._is_running:
            logger.warning(f"Agent '{self.name}' is already running. Skipping start.")
            return
        self._is_running = True
        logger.info(f"Agent '{self.name}' (ID: {self.id}) started.")

    @abstractmethod
    async def stop(self):
        """
        Asynchronously stops the agent. This method must be implemented by subclasses
        to define agent-specific shutdown logic, such as releasing resources or
        gracefully terminating operations.

        It should set `_is_running` to False.
        """
        if not self._is_running:
            logger.warning(f"Agent '{self.name}' is not running. Skipping stop.")
            return
        self._is_running = False
        logger.info(f"Agent '{self.name}' (ID: {self.id}) stopped.")

    @abstractmethod
    async def process_message(self, message: Any):
        """
        Abstract method for agents to asynchronously process incoming messages.
        Concrete agent implementations must define how they react to and
        process different types of messages.

        Args:
            message (Any): The incoming message to be processed.
        """
        logger.debug(f"Agent '{self.name}' (ID: {self.id}) received message: {message}")
        # Subclasses will add specific processing logic here.
        pass

    @abstractmethod
    async def handle_task(self, task: Dict[str, Any]):
        """
        Abstract method for agents to asynchronously handle assigned tasks.
        Concrete agent implementations must define the logic for executing
        or delegating specific tasks.

        Args:
            task (Dict[str, Any]): A dictionary containing the task details.
        """
        logger.debug(f"Agent '{self.name}' (ID: {self.id}) received task: {task.get('type')}")
        # Subclasses will add specific task handling logic here.
        pass

    @abstractmethod
    async def perform_action(self, **kwargs: Any) -> Any:
        """
        Abstract method for an agent to perform its primary, autonomous action.
        This could represent a continuous operational loop, a reaction to environmental
        changes, or the execution of a high-level goal. The specific behavior
        is defined by concrete agent implementations.

        Args:
            **kwargs (Any): Arbitrary keyword arguments that may influence the action.

        Returns:
            Any: The result of the action, if any.
        """
        logger.debug(f"Agent '{self.name}' (ID: {self.id}) performing action with kwargs: {kwargs}")
        # Subclasses will add specific action logic here.
        pass
```