```python
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable

# Assuming these imports exist relative to the project root
# For core/orchestrator.py, these imports would be absolute paths from the project root.
from core.state_manager import StateManager
from core.event_bus import EventBus, Event
from core.agent_lifecycle import AgentLifecycleManager
from agents.base_agent import BaseAgent
from specifications.constraint_engine import ConstraintEngine
from specifications.interaction_spec_validator import InteractionSpecValidator
from supervisor.control_plane import ControlPlane
from supervisor.policy_engine import PolicyEngine
from monitoring_debugging.logger_config import get_logger
from monitoring_debugging.tracer import Tracer

# Initialize logger and tracer
logger = get_logger(__name__)
tracer = Tracer()

class Orchestrator:
    """
    The central engine responsible for scheduling agent actions, managing execution flow,
    and coordinating inter-agent interactions within the Agent Orchestration Control Framework.

    This class provides explicit control, deterministic state management, and enforces
    formal specifications and policies at runtime.
    """

    def __init__(self,
                 state_manager: StateManager,
                 event_bus: EventBus,
                 agent_lifecycle_manager: AgentLifecycleManager,
                 constraint_engine: ConstraintEngine,
                 interaction_spec_validator: InteractionSpecValidator,
                 control_plane: ControlPlane,
                 policy_engine: PolicyEngine,
                 loop_interval_sec: float = 0.1):
        """
        Initializes the Orchestrator with required core components.

        Args:
            state_manager: The system's state manager.
            event_bus: The central event bus for communication.
            agent_lifecycle_manager: Manages agent creation, initialization, and termination.
            constraint_engine: Enforces formal constraints on agent actions.
            interaction_spec_validator: Validates inter-agent communication schemas and protocols.
            control_plane: Provides external control mechanisms for pausing, resuming, etc.
            policy_engine: Enforces system-wide operational policies and rules.
            loop_interval_sec: The sleep interval for the main orchestration loop (in seconds).
                               This prevents busy-waiting when no actions are pending.
        """
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.agent_lifecycle_manager = agent_lifecycle_manager
        self.constraint_engine = constraint_engine
        self.interaction_spec_validator = interaction_spec_validator
        self.control_plane = control_plane
        self.policy_engine = policy_engine
        self.loop_interval_sec = loop_interval_sec

        self._registered_agents: Dict[str, BaseAgent] = {}
        self._is_running: bool = False
        self._is_paused: bool = False
        self._orchestration_task: Optional[asyncio.Task] = None
        # Queue for agent actions requested by agents or external systems
        self._agent_action_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        logger.info("Orchestrator initialized.")
        tracer.trace("Orchestrator", "Initialized", {"loop_interval": loop_interval_sec})

        # Subscribe to relevant events from the EventBus
        self._subscribe_to_events()

    def _subscribe_to_events(self):
        """Subscribes the orchestrator to necessary events on the event bus."""
        # Agents request actions by publishing this event
        self.event_bus.subscribe("AGENT_ACTION_REQUEST", self._handle_agent_action_request)
        # Agents communicate with each other via this message type
        self.event_bus.subscribe("AGENT_MESSAGE", self._handle_agent_message)
        # Control plane or meta-agents can issue these commands
        self.event_bus.subscribe("SYSTEM_PAUSE_REQUEST", self._handle_system_pause)
        self.event_bus.subscribe("SYSTEM_RESUME_REQUEST", self._handle_system_resume)
        self.event_bus.subscribe("SYSTEM_TERMINATE_REQUEST", self._handle_system_terminate)
        logger.debug("Orchestrator subscribed to event bus events.")

    async def register_agent(self, agent: BaseAgent):
        """
        Registers an agent with the orchestrator and initializes its lifecycle.
        Updates the global state manager with the agent's initial status.

        Args:
            agent: The BaseAgent instance to register.
        """
        if agent.agent_id in self._registered_agents:
            logger.warning(f"Agent with ID '{agent.agent_id}' is already registered.")
            return

        try:
            self._registered_agents[agent.agent_id] = agent
            await self.agent_lifecycle_manager.initialize_agent(agent)
            await self.state_manager.update_agent_state(agent.agent_id, {"status": "initialized", "last_action": None})
            logger.info(f"Agent '{agent.agent_id}' registered and initialized.")
            tracer.trace("Orchestrator", "AgentRegistered", {"agent_id": agent.agent_id})
        except Exception as e:
            logger.error(f"Failed to register agent '{agent.agent_id}': {e}", exc_info=True)
            tracer.trace("Orchestrator", "AgentRegistrationFailed", {"agent_id": agent.agent_id, "error": str(e)}, "ERROR")

    async def unregister_agent(self, agent_id: str):
        """
        Unregisters an agent from the orchestrator and handles its termination.
        Removes the agent's state from the global state manager.

        Args:
            agent_id: The ID of the agent to unregister.
        """
        if agent_id not in self._registered_agents:
            logger.warning(f"Attempted to unregister non-existent agent '{agent_id}'.")
            return

        try:
            agent = self._registered_agents.pop(agent_id)
            await self.agent_lifecycle_manager.terminate_agent(agent)
            await self.state_manager.remove_agent_state(agent_id)
            logger.info(f"Agent '{agent_id}' unregistered and terminated.")
            tracer.trace("Orchestrator", "AgentUnregistered", {"agent_id": agent_id})
        except Exception as e:
            logger.error(f"Failed to unregister agent '{agent_id}': {e}", exc_info=True)
            tracer.trace("Orchestrator", "AgentUnregistrationFailed", {"agent_id": agent_id, "error": str(e)}, "ERROR")

    async def start(self):
        """
        Starts the main orchestration loop as an asynchronous task.
        Updates the system's global state to 'running'.
        """
        if self._is_running:
            logger.warning("Orchestrator is already running.")
            return

        self._is_running = True
        self._is_paused = False # Ensure not paused when starting
        self._orchestration_task = asyncio.create_task(self._orchestration_loop())
        await self.state_manager.update_system_state({"status": "running"})
        logger.info("Orchestrator started.")
        tracer.trace("Orchestrator", "Started")

        # Give a moment for the loop to start and initial tasks to queue up
        await asyncio.sleep(0.01)

    async def stop(self):
        """
        Stops the main orchestration loop and performs necessary cleanup.
        Awaits the completion of the orchestration task.
        """
        if not self._is_running:
            logger.warning("Orchestrator is not running.")
            return

        self._is_running = False
        if self._orchestration_task:
            # Signal the loop to terminate and await its completion
            await self._orchestration_task
            self._orchestration_task = None

        await self.state_manager.update_system_state({"status": "stopped"})
        logger.info("Orchestrator stopped.")
        tracer.trace("Orchestrator", "Stopped")

    async def _orchestration_loop(self):
        """
        The main asynchronous loop for orchestrating agent actions.
        It continuously checks for pending actions, applies rules and policies,
        and then executes the actions.
        """
        logger.info("Orchestration loop started.")
        while self._is_running:
            try:
                if self._is_paused:
                    # If paused, just sleep and check again later
                    await asyncio.sleep(self.loop_interval_sec)
                    continue

                # 1. Process pending agent actions from the queue
                if not self._agent_action_queue.empty():
                    action_request = await self._agent_action_queue.get()
                    await self._execute_agent_action(action_request)
                    self._agent_action_queue.task_done()
                else:
                    # If no actions in the queue, give other async tasks a chance
                    # and prevent the loop from consuming 100% CPU
                    await asyncio.sleep(self.loop_interval_sec)

                # TODO: Future enhancements for the orchestration logic:
                # - Implement more sophisticated scheduling algorithms (e.g., priority-based, resource-aware).
                # - Trigger meta-agent decisions based on system state or emergent behaviors.
                # - Actively poll for agent intentions or state changes if not event-driven.

            except asyncio.CancelledError:
                logger.info("Orchestration loop cancelled.")
                break # Exit loop cleanly
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}", exc_info=True)
                tracer.trace("Orchestrator", "LoopError", {"error": str(e)}, "ERROR")

        logger.info("Orchestration loop terminated.")

    async def _handle_agent_action_request(self, event: Event):
        """
        Handles an 'AGENT_ACTION_REQUEST' event by placing the action into the internal queue.
        This allows the orchestrator to process actions sequentially and apply controls.
        """
        payload = event.payload
        agent_id = payload.get("agent_id")
        action = payload.get("action")

        if not all([agent_id, action]):
            logger.error(f"Invalid AGENT_ACTION_REQUEST payload: Missing agent_id or action. Payload: {payload}")
            tracer.trace("Orchestrator", "InvalidActionRequest", payload, "ERROR")
            return

        logger.debug(f"Received action request from '{agent_id}': {action}")
        tracer.trace("Orchestrator", "ActionRequestReceived", payload)

        # Basic validation: ensure the requesting agent is known to the orchestrator
        if agent_id not in self._registered_agents:
            logger.warning(f"Action request from unregistered agent '{agent_id}'. Ignoring.")
            tracer.trace("Orchestrator", "UnregisteredAgentAction", payload, "WARNING")
            return

        # Add the action request to the queue for orchestrated execution
        await self._agent_action_queue.put(payload)

    async def _execute_agent_action(self, action_request: Dict[str, Any]):
        """
        Executes a single agent action after applying formal specification checks
        and policy enforcement. Updates system and agent states accordingly.
        """
        agent_id = action_request.get("agent_id")
        action_name = action_request.get("action")
        target_id = action_request.get("target_id") # E.g., another agent, a tool ID
        action_details = action_request.get("details", {}) # Specific parameters for the action
        original_message = action_request.get("message", {}) # If the action is specifically sending a message

        logger.debug(f"Attempting to execute action '{action_name}' for agent '{agent_id}'...")
        tracer.trace("Orchestrator", "ExecutingAction", action_request)

        agent = self._registered_agents.get(agent_id)
        if not agent:
            logger.error(f"Agent '{agent_id}' not found for action execution.")
            tracer.trace("Orchestrator", "AgentNotFoundForAction", action_request, "ERROR")
            return

        try:
            # 1. Formal Specification Enforcement (Pre-condition checks)
            # Validate if the action itself adheres to defined constraints
            if not self.constraint_engine.validate_action(agent_id, action_name, action_details):
                raise ValueError(f"Action '{action_name}' by '{agent_id}' violates defined constraints.")

            # If the action is a message, validate its schema
            if action_name == "send_message" and original_message:
                if not self.interaction_spec_validator.validate_message(original_message):
                    raise ValueError(f"Message within action '{action_name}' from '{agent_id}' to '{target_id}' violates communication schema.")

            # 2. Policy Enforcement
            # Authorize the action based on global policies set by the policy engine
            if not await self.policy_engine.authorize_action(agent_id, action_name, action_details):
                raise PermissionError(f"Action '{action_name}' by '{agent_id}' denied by policy.")

            # 3. Execute the action by delegating to the agent
            # Agents are expected to implement 'execute_orchestrated_action' to handle directives
            result = await agent.execute_orchestrated_action(action_name, action_details, target_id)

            # 4. State Management: Update global and agent-specific states
            await self.state_manager.update_agent_state(agent_id, {"last_action": action_name, "status": "active"})
            if target_id:
                await self.state_manager.update_agent_state(target_id, {"last_interaction_from": agent_id})
            await self.state_manager.update_system_state({"last_agent_action": {"agent_id": agent_id, "action": action_name}})

            # 5. Publish completion event
            await self.event_bus.publish(
                Event("AGENT_ACTION_COMPLETED", {
                    "agent_id": agent_id,
                    "action": action_name,
                    "result": result,
                    "timestamp": time.time()
                })
            )
            logger.info(f"Action '{action_name}' by '{agent_id}' completed successfully.")
            tracer.trace("Orchestrator", "ActionCompleted", {"agent_id": agent_id, "action": action_name, "result": str(result)})

        except (ValueError, PermissionError) as e:
            # Specific errors due to policy or constraint violations
            logger.warning(f"Action '{action_name}' by '{agent_id}' failed due to validation/policy: {e}")
            await self.event_bus.publish(
                Event("AGENT_ACTION_FAILED", {
                    "agent_id": agent_id,
                    "action": action_name,
                    "reason": str(e),
                    "timestamp": time.time()
                })
            )
            tracer.trace("Orchestrator", "ActionFailed", {"agent_id": agent_id, "action": action_name, "error": str(e)}, "WARNING")
            await self.state_manager.update_agent_state(agent_id, {"status": "failed_action", "error": str(e)})

        except Exception as e:
            # Catch any other unexpected errors during action execution
            logger.error(f"Unexpected error during action '{action_name}' by '{agent_id}': {e}", exc_info=True)
            await self.event_bus.publish(
                Event("AGENT_ACTION_FAILED", {
                    "agent_id": agent_id,
                    "action": action_name,
                    "reason": f"Unexpected error: {e}",
                    "timestamp": time.time()
                })
            )
            tracer.trace("Orchestrator", "ActionFailed", {"agent_id": agent_id, "action": action_name, "error": str(e)}, "ERROR")
            await self.state_manager.update_agent_state(agent_id, {"status": "error", "error": str(e)})

    async def _handle_agent_message(self, event: Event):
        """
        Handles an 'AGENT_MESSAGE' event, performing validation and policy checks
        on inter-agent communication. Updates relevant agent states.
        """
        payload = event.payload
        sender_id = payload.get("sender_id")
        receiver_id = payload.get("receiver_id")
        message_content = payload.get("content")

        if not all([sender_id, receiver_id, message_content]):
            logger.error(f"Invalid AGENT_MESSAGE payload: Missing sender, receiver, or content. Payload: {payload}")
            tracer.trace("Orchestrator", "InvalidAgentMessage", payload, "ERROR")
            return

        logger.debug(f"Received message from '{sender_id}' to '{receiver_id}'.")
        tracer.trace("Orchestrator", "AgentMessageReceived", payload)

        try:
            # 1. Validate message schema using the interaction spec validator
            if not self.interaction_spec_validator.validate_message(payload):
                raise ValueError(f"Message from '{sender_id}' to '{receiver_id}' violates communication schema.")

            # 2. Policy Enforcement: Check if this communication is allowed
            if not await self.policy_engine.authorize_communication(sender_id, receiver_id, message_content):
                raise PermissionError(f"Communication from '{sender_id}' to '{receiver_id}' denied by policy.")

            # 3. Update state for sender and receiver
            await self.state_manager.update_agent_state(sender_id, {"last_sent_message_to": receiver_id})
            await self.state_manager.update_agent_state(receiver_id, {"last_received_message_from": sender_id})
            await self.state_manager.update_system_state({"last_message": {"sender": sender_id, "receiver": receiver_id, "timestamp": time.time()}})

            # The actual message delivery to the receiver agent is typically handled by
            # the `communication_layer.py` which would listen for AGENT_MESSAGE events.
            # The orchestrator's role here is primarily validation and state update.
            logger.info(f"Message from '{sender_id}' to '{receiver_id}' processed by orchestrator (validated and state updated).")
            tracer.trace("Orchestrator", "AgentMessageProcessed", payload)

        except (ValueError, PermissionError) as e:
            logger.warning(f"Message from '{sender_id}' to '{receiver_id}' failed validation/policy: {e}")
            tracer.trace("Orchestrator", "AgentMessageFailed", {"sender": sender_id, "receiver": receiver_id, "error": str(e)}, "WARNING")
            # Publish an event indicating message delivery failure
            await self.event_bus.publish(
                Event("MESSAGE_DELIVERY_FAILED", {
                    "sender_id": None, # Orchestrator or system as sender of error
                    "receiver_id": sender_id, # Notify the original sender
                    "original_message": payload,
                    "reason": str(e),
                    "timestamp": time.time()
                })
            )
        except Exception as e:
            logger.error(f"Unexpected error handling message from '{sender_id}' to '{receiver_id}': {e}", exc_info=True)
            tracer.trace("Orchestrator", "AgentMessageError", {"sender": sender_id, "receiver": receiver_id, "error": str(e)}, "ERROR")


    async def _handle_system_pause(self, event: Event):
        """
        Handles a 'SYSTEM_PAUSE_REQUEST' event, pausing the orchestration loop.
        Updates the system's global state to 'paused'.
        """
        if not self._is_paused:
            self._is_paused = True
            await self.state_manager.update_system_state({"status": "paused"})
            logger.info("Orchestrator paused by system request.")
            tracer.trace("Orchestrator", "SystemPaused", event.payload)
        else:
            logger.warning("Orchestrator already paused. Ignoring pause request.")

    async def _handle_system_resume(self, event: Event):
        """
        Handles a 'SYSTEM_RESUME_REQUEST' event, resuming the orchestration loop.
        Updates the system's global state to 'running'.
        """
        if self._is_paused:
            self._is_paused = False
            await self.state_manager.update_system_state({"status": "running"})
            logger.info("Orchestrator resumed by system request.")
            tracer.trace("Orchestrator", "SystemResumed", event.payload)
        else:
            logger.warning("Orchestrator is not paused. Ignoring resume request.")

    async def _handle_system_terminate(self, event: Event):
        """
        Handles a 'SYSTEM_TERMINATE_REQUEST' event, initiating a graceful shutdown
        of the orchestrator.
        """
        logger.info("Orchestrator received termination request. Stopping...")
        await self.stop()
        tracer.trace("Orchestrator", "SystemTerminated", event.payload)

    @property
    def registered_agents(self) -> Dict[str, BaseAgent]:
        """Returns a read-only dictionary of currently registered agents."""
        return self._registered_agents

    @property
    def is_running(self) -> bool:
        """Returns True if the orchestrator loop is actively running."""
        return self._is_running

    @property
    def is_paused(self) -> bool:
        """Returns True if the orchestrator is currently in a paused state."""
        return self._is_paused
```