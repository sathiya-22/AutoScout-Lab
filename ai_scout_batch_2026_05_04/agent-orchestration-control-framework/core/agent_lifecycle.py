```python
import logging
import importlib
import uuid
from typing import Dict, Any, Type, Optional

# Assuming these modules exist and provide the necessary interfaces/classes.
# In a real project, these would be uncommented and the Mock versions removed.
# from core.state_manager import StateManager
# from core.event_bus import EventBus
# from agents.base_agent import BaseAgent
# from config.settings import AGENT_CONFIGS_PATH, LOGGING_LEVEL
# from config.logger_config import setup_logging # For centralized logging setup
import yaml

# --- Mock Implementations for development/standalone testing ---
# These mocks simulate the dependencies to allow agent_lifecycle.py to be
# developed and tested in isolation before the full framework is complete.
# In the complete project, replace these with actual imports.

class MockStateManager:
    """A mock StateManager for testing AgentLifecycleManager."""
    def __init__(self):
        self._agent_states: Dict[str, str] = {}
        self._agent_instances: Dict[str, Any] = {}
        self._agent_internal_states: Dict[str, Dict[str, Any]] = {}

    def add_agent(self, agent_id: str, initial_status: str, agent_instance: Any):
        self._agent_states[agent_id] = initial_status
        self._agent_instances[agent_id] = agent_instance
        self._agent_internal_states[agent_id] = {}
        logging.info(f"[MockStateManager] Added agent '{agent_id}' with status '{initial_status}'")

    def update_agent_state(self, agent_id: str, new_status: str):
        if agent_id not in self._agent_states:
            raise ValueError(f"Agent '{agent_id}' not found in state manager.")
        logging.info(f"[MockStateManager] Agent '{agent_id}' state: {self._agent_states[agent_id]} -> {new_status}")
        self._agent_states[agent_id] = new_status

    def get_agent_state(self, agent_id: str) -> Optional[str]:
        return self._agent_states.get(agent_id)
    
    def get_agent_instance(self, agent_id: str) -> Optional[Any]:
        # AgentLifecycleManager will manage instances, StateManager primarily state/metadata.
        # This mock provides it for completeness, though LifecycleManager uses its own cache.
        return self._agent_instances.get(agent_id)

    def update_agent_internal_state(self, agent_id: str, state_data: Dict[str, Any]):
        if agent_id not in self._agent_internal_states:
            raise ValueError(f"Agent '{agent_id}' not found for internal state update.")
        self._agent_internal_states[agent_id].update(state_data)
        logging.debug(f"[MockStateManager] Updated internal state for '{agent_id}': {state_data}")

    def get_agent_internal_state(self, agent_id: str) -> Dict[str, Any]:
        return self._agent_internal_states.get(agent_id, {})

    def remove_agent(self, agent_id: str):
        if agent_id in self._agent_states:
            del self._agent_states[agent_id]
        if agent_id in self._agent_instances:
            del self._agent_instances[agent_id]
        if agent_id in self._agent_internal_states:
            del self._agent_internal_states[agent_id]
        logging.info(f"[MockStateManager] Removed agent '{agent_id}'")

StateManager = MockStateManager # Assign Mock to actual name for this file

class MockEventBus:
    """A mock EventBus for testing AgentLifecycleManager."""
    def publish(self, event_type: str, payload: Dict[str, Any]):
        logging.info(f"[MockEventBus] Published event '{event_type}' with payload: {payload}")

EventBus = MockEventBus # Assign Mock to actual name for this file

class MockBaseAgent:
    """A mock BaseAgent for testing AgentLifecycleManager."""
    def __init__(self, agent_id: str, config: Dict[str, Any], state_manager: StateManager, event_bus: EventBus):
        self.agent_id = agent_id
        self.config = config
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.logger = logging.getLogger(f"MockAgent-{agent_id}")
        self.logger.setLevel(logging.INFO) # In a real app, use global LOGGING_LEVEL

    def initialize(self):
        self.logger.info(f"MockAgent {self.agent_id} initializing with config: {self.config}")

    def suspend(self):
        self.logger.info(f"MockAgent {self.agent_id} suspending.")

    def resume(self):
        self.logger.info(f"MockAgent {self.agent_id} resuming.")

    def terminate(self):
        self.logger.info(f"MockAgent {self.agent_id} terminating.")

BaseAgent = MockBaseAgent # Assign Mock to actual name for this file

class MockSettings:
    """A mock Settings class to provide config paths and logging level."""
    AGENT_CONFIGS_PATH = "config/agent_configs.yaml" # Placeholder path
    LOGGING_LEVEL = logging.INFO

Settings = MockSettings # Assign Mock to actual name for this file

# Set up basic logging for the mock environment
logging.basicConfig(level=Settings.LOGGING_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Agent States (Consider moving to a shared core/constants.py or core/enums.py) ---
class AgentStatus:
    """Defines standard lifecycle states for agents."""
    PENDING_CREATION = "PENDING_CREATION"
    CREATED = "CREATED"
    INITIALIZING = "INITIALIZING"
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    SUSPENDED = "SUSPENDED"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"
    ERROR = "ERROR"


class AgentLifecycleManager:
    """
    Manages the complete lifecycle of AI agents within the orchestration framework.
    This includes creating, initializing, suspending, resuming, and terminating agents.

    It integrates with the StateManager to maintain canonical agent states and
    the EventBus to publish lifecycle-related events for monitoring and reactive logic.
    """

    def __init__(self, state_manager: StateManager, event_bus: EventBus):
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(Settings.LOGGING_LEVEL) # Use global logging level from config.settings

        self.agent_configurations: Dict[str, Any] = self._load_agent_configurations()
        # Stores active, instantiated agent objects, mapping agent_id to BaseAgent instance.
        # This cache helps quickly retrieve agent instances for lifecycle operations.
        self.active_agents: Dict[str, BaseAgent] = {}

    def _load_agent_configurations(self) -> Dict[str, Any]:
        """
        Loads agent configurations from the YAML file specified in the global settings.

        Returns:
            Dict[str, Any]: A dictionary of agent configurations, keyed by config name.
                            Returns an empty dictionary if the file is not found or parsing fails.
        """
        try:
            with open(Settings.AGENT_CONFIGS_PATH, 'r') as f:
                config_data = yaml.safe_load(f)
                return config_data if config_data is not None else {}
        except FileNotFoundError:
            self.logger.warning(
                f"Agent configuration file not found at '{Settings.AGENT_CONFIGS_PATH}'. "
                "Starting AgentLifecycleManager with no predefined agent configurations."
            )
            return {}
        except yaml.YAMLError as e:
            self.logger.error(
                f"Error parsing agent configuration file '{Settings.AGENT_CONFIGS_PATH}': {e}"
            )
            return {}
        except Exception as e:
            self.logger.error(
                f"Unexpected error loading agent configurations from '{Settings.AGENT_CONFIGS_PATH}': {e}",
                exc_info=True
            )
            return {}

    def _get_agent_class(self, class_path: str) -> Type[BaseAgent]:
        """
        Dynamically imports and returns an agent class given its fully qualified path.

        Args:
            class_path (str): The full path to the agent class (e.g., "agents.example_agent.ExampleAgent").

        Returns:
            Type[BaseAgent]: The dynamically loaded agent class.

        Raises:
            ValueError: If the class path is invalid, the module/class cannot be found,
                        or the loaded class is not a subclass of BaseAgent.
        """
        try:
            module_name, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name)
            if not issubclass(agent_class, BaseAgent):
                raise TypeError(f"Class '{class_path}' is not a subclass of BaseAgent.")
            return agent_class
        except (ImportError, AttributeError, TypeError) as e:
            self.logger.error(f"Failed to load agent class '{class_path}': {e}", exc_info=True)
            raise ValueError(
                f"Invalid agent class path or class not found/incorrect type: '{class_path}'"
            ) from e

    def create_agent(self, agent_config_name: str, agent_id: Optional[str] = None, initial_agent_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Creates, instantiates, and initializes a new agent based on a named configuration.
        The agent is registered with the StateManager and its lifecycle events are published.

        Args:
            agent_config_name (str): The key in `agent_configs.yaml` that specifies the agent's setup.
            agent_id (Optional[str]): A unique identifier for the new agent. If None, a UUID is generated.
            initial_agent_state (Optional[Dict[str, Any]]): Initial internal state data for the agent
                                                            to be stored and managed by the StateManager.

        Returns:
            str: The unique ID of the created and initialized agent.

        Raises:
            ValueError: If the agent configuration is not found, invalid, the agent ID already exists,
                        or if dynamic class loading fails.
            Exception: For any other unforeseen errors during agent instantiation or initialization.
        """
        if agent_config_name not in self.agent_configurations:
            self.logger.error(f"Agent configuration '{agent_config_name}' not found in loaded configurations.")
            raise ValueError(f"Agent configuration '{agent_config_name}' not found.")

        config_entry = self.agent_configurations[agent_config_name]
        agent_class_path = config_entry.get("class_path")
        if not agent_class_path:
            self.logger.error(f"Agent configuration '{agent_config_name}' is missing the 'class_path' attribute.")
            raise ValueError(f"Agent configuration '{agent_config_name}' missing 'class_path'.")

        agent_id = agent_id if agent_id else str(uuid.uuid4())
        
        # Prevent creation if an agent with the same ID is already active
        if agent_id in self.active_agents:
            self.logger.error(f"Agent with ID '{agent_id}' is already active.")
            raise ValueError(f"Agent with ID '{agent_id}' already exists.")

        self.logger.info(f"Initiating creation of agent '{agent_id}' (Type: '{agent_config_name}')...")
        
        try:
            # 1. Dynamically load the agent class
            AgentClass = self._get_agent_class(agent_class_path)

            # 2. Instantiate the agent
            # Pass core framework components and specific agent parameters to the agent's constructor
            agent_instance = AgentClass(
                agent_id=agent_id,
                config=config_entry.get("parameters", {}), # Pass only agent-specific parameters
                state_manager=self.state_manager,
                event_bus=self.event_bus
            )
            self.active_agents[agent_id] = agent_instance # Add to internal cache

            # 3. Register the agent with the StateManager
            # Initial status is CREATED, pending its internal initialization
            self.state_manager.add_agent(agent_id, AgentStatus.CREATED, agent_instance)
            if initial_agent_state:
                self.state_manager.update_agent_internal_state(agent_id, initial_agent_state)

            # 4. Call the agent's custom initialization logic
            self.state_manager.update_agent_state(agent_id, AgentStatus.INITIALIZING)
            agent_instance.initialize() # Assumes BaseAgent defines an `initialize` method
            self.state_manager.update_agent_state(agent_id, AgentStatus.INITIALIZED)

            self.logger.info(f"Agent '{agent_id}' (Type: '{agent_config_name}') created and initialized successfully.")
            self.event_bus.publish(
                "agent_lifecycle.created",
                {"agent_id": agent_id, "agent_type": agent_config_name, "status": AgentStatus.INITIALIZED}
            )
            
            return agent_id
        except Exception as e:
            # Ensure proper cleanup if creation fails at any stage
            self.logger.error(
                f"Failed to create or initialize agent '{agent_id}' (Type: '{agent_config_name}'): {e}",
                exc_info=True
            )
            # Attempt to update state to ERROR if it was partially added
            if self.state_manager.get_agent_state(agent_id):
                self.state_manager.update_agent_state(agent_id, AgentStatus.ERROR)
            self.event_bus.publish(
                "agent_lifecycle.error",
                {"agent_id": agent_id, "agent_type": agent_config_name, "error": str(e), "status": AgentStatus.ERROR}
            )
            # Remove from active agents cache and state manager to prevent orphaned entries
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]
            if self.state_manager.get_agent_state(agent_id):
                self.state_manager.remove_agent(agent_id)
            raise # Re-raise the exception to signal failure to the caller

    def suspend_agent(self, agent_id: str) -> None:
        """
        Suspends a running or initialized agent, pausing its operations.
        The agent's state is updated, and a `suspended` event is published.

        Args:
            agent_id (str): The ID of the agent to suspend.

        Raises:
            ValueError: If the agent is not found or is not in a suspendable state.
            Exception: For errors occurring within the agent's custom `suspend` method.
        """
        self.logger.info(f"Attempting to suspend agent '{agent_id}'...")
        try:
            agent_instance = self._get_agent_instance(agent_id)
            current_status = self.state_manager.get_agent_state(agent_id)

            if current_status not in [AgentStatus.INITIALIZED, AgentStatus.RUNNING]:
                self.logger.warning(
                    f"Agent '{agent_id}' is in state '{current_status}', which is not suspendable. "
                    "Only agents in INITIALIZED or RUNNING state can be suspended."
                )
                raise ValueError(f"Agent '{agent_id}' not in a suspendable state (current: {current_status}).")

            agent_instance.suspend() # Assumes BaseAgent defines a `suspend` method
            self.state_manager.update_agent_state(agent_id, AgentStatus.SUSPENDED)
            self.logger.info(f"Agent '{agent_id}' suspended successfully.")
            self.event_bus.publish(
                "agent_lifecycle.suspended",
                {"agent_id": agent_id, "status": AgentStatus.SUSPENDED}
            )
        except ValueError:
            raise # Re-raise specific ValueErrors (e.g., agent not found, invalid state)
        except Exception as e:
            # Transition to error state if suspend fails
            self.state_manager.update_agent_state(agent_id, AgentStatus.ERROR)
            self.logger.error(f"Error suspending agent '{agent_id}': {e}", exc_info=True)
            self.event_bus.publish(
                "agent_lifecycle.error",
                {"agent_id": agent_id, "error": str(e), "status": AgentStatus.ERROR}
            )
            raise

    def resume_agent(self, agent_id: str) -> None:
        """
        Resumes a suspended agent, allowing it to continue its operations.
        The agent's state is updated, and a `resumed` event is published.

        Args:
            agent_id (str): The ID of the agent to resume.

        Raises:
            ValueError: If the agent is not found or is not in a resumable state.
            Exception: For errors occurring within the agent's custom `resume` method.
        """
        self.logger.info(f"Attempting to resume agent '{agent_id}'...")
        try:
            agent_instance = self._get_agent_instance(agent_id)
            current_status = self.state_manager.get_agent_state(agent_id)

            if current_status != AgentStatus.SUSPENDED:
                self.logger.warning(
                    f"Agent '{agent_id}' is in state '{current_status}', which is not resumable. "
                    "Only agents in SUSPENDED state can be resumed."
                )
                raise ValueError(f"Agent '{agent_id}' not in a resumable state (current: {current_status}).")

            agent_instance.resume() # Assumes BaseAgent defines a `resume` method
            # After resuming, an agent typically transitions to RUNNING.
            self.state_manager.update_agent_state(agent_id, AgentStatus.RUNNING)
            self.logger.info(f"Agent '{agent_id}' resumed successfully.")
            self.event_bus.publish(
                "agent_lifecycle.resumed",
                {"agent_id": agent_id, "status": AgentStatus.RUNNING}
            )
        except ValueError:
            raise
        except Exception as e:
            self.state_manager.update_agent_state(agent_id, AgentStatus.ERROR)
            self.logger.error(f"Error resuming agent '{agent_id}': {e}", exc_info=True)
            self.event_bus.publish(
                "agent_lifecycle.error",
                {"agent_id": agent_id, "error": str(e), "status": AgentStatus.ERROR}
            )
            raise

    def terminate_agent(self, agent_id: str) -> None:
        """
        Terminates an agent, shutting it down gracefully and removing it from active management.
        The agent's state is updated, a `terminated` event is published, and it's removed
        from the internal cache and the StateManager.

        Args:
            agent_id (str): The ID of the agent to terminate.

        Raises:
            ValueError: If the agent is not found.
            Exception: For errors occurring within the agent's custom `terminate` method.
        """
        self.logger.info(f"Attempting to terminate agent '{agent_id}'...")
        try:
            agent_instance = self._get_agent_instance(agent_id)
            current_status = self.state_manager.get_agent_state(agent_id)

            if current_status == AgentStatus.TERMINATED:
                self.logger.warning(f"Agent '{agent_id}' is already in TERMINATED state. No action taken.")
                return # Idempotent: already terminated is a successful outcome

            self.state_manager.update_agent_state(agent_id, AgentStatus.TERMINATING)
            agent_instance.terminate() # Assumes BaseAgent defines a `terminate` method
            self.state_manager.update_agent_state(agent_id, AgentStatus.TERMINATED)

            # Clean up internal tracking and state manager entries
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]
            self.state_manager.remove_agent(agent_id)

            self.logger.info(f"Agent '{agent_id}' terminated successfully.")
            self.event_bus.publish(
                "agent_lifecycle.terminated",
                {"agent_id": agent_id, "status": AgentStatus.TERMINATED}
            )
        except ValueError:
            raise
        except Exception as e:
            # Even if termination fails, we attempt to mark it as ERROR and remove
            self.state_manager.update_agent_state(agent_id, AgentStatus.ERROR)
            self.logger.error(f"Error during termination of agent '{agent_id}': {e}", exc_info=True)
            self.event_bus.publish(
                "agent_lifecycle.error",
                {"agent_id": agent_id, "error": str(e), "status": AgentStatus.ERROR}
            )
            # Still attempt to remove from cache and state manager to avoid orphaned entries
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]
            self.state_manager.remove_agent(agent_id)
            raise

    def get_agent(self, agent_id: str) -> BaseAgent:
        """
        Retrieves an active agent instance by its ID from the internal cache.

        Args:
            agent_id (str): The ID of the agent to retrieve.

        Returns:
            BaseAgent: The agent instance.

        Raises:
            ValueError: If the agent is not found in the active agents cache.
        """
        return self._get_agent_instance(agent_id)

    def _get_agent_instance(self, agent_id: str) -> BaseAgent:
        """
        Helper method to retrieve an agent instance from the internal `active_agents` cache.

        Args:
            agent_id (str): The ID of the agent to retrieve.

        Returns:
            BaseAgent: The agent instance.

        Raises:
            ValueError: If the agent is not found in the active agents cache.
        """
        agent_instance = self.active_agents.get(agent_id)
        if not agent_instance:
            self.logger.error(f"Agent with ID '{agent_id}' not found in active agents cache.")
            raise ValueError(f"Agent with ID '{agent_id}' not found.")
        return agent_instance
```