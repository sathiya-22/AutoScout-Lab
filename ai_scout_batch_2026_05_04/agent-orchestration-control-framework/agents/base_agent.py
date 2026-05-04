import abc
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING
import uuid

# Use TYPE_CHECKING for circular dependency resolution during runtime
if TYPE_CHECKING:
    from core.event_bus import EventBus
    from protocols.message_schemas import BaseMessage

class BaseAgent(abc.ABC):
    """
    An abstract base class defining the common interface and lifecycle methods for all agents
    within the Agent Orchestration Control Framework.

    All concrete agents must inherit from this class and implement its abstract methods.
    This class provides core functionalities like unique identification, name, configuration
    management, logging, and interaction with the central EventBus.
    """

    def __init__(self, name: str, event_bus: 'EventBus', config: Optional[Dict[str, Any]] = None):
        """
        Initializes the BaseAgent with a unique ID, a human-readable name,
        a reference to the central EventBus, and an optional configuration dictionary.

        Args:
            name (str): The human-readable name of the agent.
            event_bus (EventBus): The central event bus instance for inter-agent communication.
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters specific
                                               to this agent instance. Defaults to an empty dict.
        """
        self._agent_id: str = str(uuid.uuid4())  # Unique identifier for each agent instance
        self._name: str = name
        self._event_bus: 'EventBus' = event_bus
        self._config: Dict[str, Any] = config if config is not None else {}
        
        # Agent's internal, local state view. The canonical state for all agents
        # is managed by the StateManager. This is for quick local access/cache.
        self._internal_state: Dict[str, Any] = {"status": "initialized"}

        self.logger = logging.getLogger(f"Agent.{self.name}")
        self.logger.info(f"Agent '{self.name}' ({self.agent_id}) initialized.")

    @property
    def agent_id(self) -> str:
        """Returns the unique identifier of the agent."""
        return self._agent_id

    @property
    def name(self) -> str:
        """Returns the human-readable name of the agent."""
        return self._name

    @property
    def config(self) -> Dict[str, Any]:
        """Returns the immutable configuration dictionary of the agent."""
        return self._config

    @abc.abstractmethod
    async def initialize(self) -> None:
        """
        Abstract method for agent-specific initialization logic.
        This method should be called once after the agent has been created and registered
        with the framework. It should handle setting up resources, performing initial
        data loads, subscribing to initial topics on the EventBus, etc.
        """
        self.logger.debug(f"Agent '{self.name}' ({self.agent_id}) initializing...")
        # Concrete agents must implement their specific initialization logic here.

    @abc.abstractmethod
    async def run(self) -> None:
        """
        Abstract method representing the main execution logic or a single decision cycle of the agent.
        The Orchestrator will typically call this method repeatedly or trigger it based on events
        to drive the agent's behavior. This method should encapsulate the agent's core task or
        decision-making process.
        """
        self.logger.debug(f"Agent '{self.name}' ({self.agent_id}) running main logic cycle...")
        # Concrete agents must implement their core operational logic here.

    @abc.abstractmethod
    async def on_message(self, topic: str, message: 'BaseMessage') -> None:
        """
        Abstract method to handle incoming messages directed to this agent or subscribed topics.
        The EventBus will route messages to this method based on the agent's subscriptions.

        Args:
            topic (str): The topic the message was published on.
            message (BaseMessage): The incoming message object, expected to conform to a
                                   defined schema from protocols.message_schemas.
        """
        self.logger.debug(f"Agent '{self.name}' ({self.agent_id}) received message on topic '{topic}'.")
        # Concrete agents must implement specific message handling logic here.

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """
        Abstract method for agent-specific cleanup logic.
        This method should be called when the agent is being terminated or suspended.
        It should handle releasing resources, unsubscribing from topics, persisting state, etc.
        """
        self.logger.debug(f"Agent '{self.name}' ({self.agent_id}) shutting down...")
        # Concrete agents must implement their specific cleanup logic here.

    async def _send_message(self, topic: str, message: 'BaseMessage') -> None:
        """
        Internal utility method to send a message via the central EventBus.
        This method serializes the message and publishes it to the specified topic.

        Args:
            topic (str): The topic to publish the message to.
            message (BaseMessage): The message object to be sent. It must be an instance
                                   of a class inheriting from BaseMessage.
        """
        try:
            # Assuming BaseMessage is a Pydantic model with a .model_dump_json() method
            message_payload = message.model_dump_json()
            await self._event_bus.publish(topic, message)
            self.logger.debug(f"Agent '{self.name}' ({self.agent_id}) published message "
                              f"to topic '{topic}': {message_payload}")
        except AttributeError:
            self.logger.error(f"Message '{message.__class__.__name__}' is not a Pydantic model or "
                              f"does not have 'model_dump_json' method. Publishing raw message.", exc_info=True)
            await self._event_bus.publish(topic, message)
        except Exception as e:
            self.logger.error(f"Error sending message from agent '{self.name}' ({self.agent_id}) "
                              f"to topic '{topic}': {e}", exc_info=True)
            # Depending on the system's robustness requirements, one might re-raise,
            # queue for retry, or notify a supervisor.

    async def _subscribe_to_topic(self, topic: str) -> None:
        """
        Internal utility method to subscribe the agent to a specific topic on the EventBus.
        Messages published on this topic will trigger the agent's `on_message` method.

        Args:
            topic (str): The topic string to subscribe to.
        """
        try:
            # The EventBus's subscribe method is expected to register a callback.
            # We bind self.on_message as the handler for messages on this topic.
            await self._event_bus.subscribe(topic, self.on_message)
            self.logger.info(f"Agent '{self.name}' ({self.agent_id}) subscribed to topic '{topic}'.")
        except Exception as e:
            self.logger.error(f"Error subscribing agent '{self.name}' ({self.agent_id}) "
                              f"to topic '{topic}': {e}", exc_info=True)
            raise  # Re-raise to indicate a critical setup failure.

    async def _unsubscribe_from_topic(self, topic: str) -> None:
        """
        Internal utility method to unsubscribe the agent from a specific topic on the EventBus.

        Args:
            topic (str): The topic string to unsubscribe from.
        """
        try:
            await self._event_bus.unsubscribe(topic, self.on_message)
            self.logger.info(f"Agent '{self.name}' ({self.agent_id}) unsubscribed from topic '{topic}'.")
        except Exception as e:
            self.logger.error(f"Error unsubscribing agent '{self.name}' ({self.agent_id}) "
                              f"from topic '{topic}': {e}", exc_info=True)
            # Log and continue, as unsubscription errors might not always be critical.

    def _update_internal_state(self, key: str, value: Any) -> None:
        """
        Updates a key-value pair in the agent's local internal state dictionary.
        This is for local state management; for canonical, system-wide state,
        agents should interact with the StateManager.

        Args:
            key (str): The key in the internal state dictionary to update.
            value (Any): The new value to set for the given key.
        """
        self._internal_state[key] = value
        self.logger.debug(f"Agent '{self.name}' ({self.agent_id}) updated internal state: {key} = {value}")

    def _get_internal_state(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a value from the agent's local internal state dictionary.

        Args:
            key (str): The key whose value is to be retrieved.
            default (Any, optional): The default value to return if the key is not found.
                                     Defaults to None.

        Returns:
            Any: The value associated with the key, or the default value if the key is not found.
        """
        return self._internal_state.get(key, default)<ctrl63>