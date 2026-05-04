import logging
from typing import Dict, Any

# Assuming these relative imports are correct based on the directory structure
# BaseAgent provides the common interface and lifecycle methods for all agents.
from .base_agent import BaseAgent
# ToolRegistry manages and provides access to external tools.
from .tool_registry import ToolRegistry

# Initialize a logger for this agent.
# In a full system, this would be configured via monitoring_debugging/logger_config.py
logger = logging.getLogger(__name__)
if not logger.handlers: # Basic configuration if run standalone for testing purposes
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ExampleAgentWithTool(BaseAgent):
    """
    An example agent that demonstrates how to utilize a registered tool to perform a task.
    This agent will attempt to use a 'calculator' tool to perform an arithmetic operation,
    emulating a simple task like data processing or calculation.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], tool_registry: ToolRegistry):
        """
        Initializes the ExampleAgentWithTool.

        Args:
            agent_id (str): A unique identifier for the agent.
            config (Dict[str, Any]): Configuration parameters specific to this agent,
                                      e.g., operation type, numbers for calculation.
            tool_registry (ToolRegistry): The central registry for accessing available tools.
        """
        super().__init__(agent_id, config)
        self.tool_registry = tool_registry
        self.calculator_tool = None  # Placeholder for the actual tool function
        logger.info(f"[{self.agent_id}] Initialized with config: {config}")

    async def setup(self) -> None:
        """
        Performs asynchronous setup for the agent.
        This includes retrieving the required tool ('calculator') from the ToolRegistry.
        """
        logger.info(f"[{self.agent_id}] Starting setup phase...")
        try:
            # Attempt to retrieve the 'calculator' tool from the registry.
            # We assume get_tool is an async method of ToolRegistry.
            self.calculator_tool = await self.tool_registry.get_tool("calculator")
            if self.calculator_tool:
                logger.info(f"[{self.agent_id}] Successfully retrieved 'calculator' tool.")
            else:
                logger.warning(f"[{self.agent_id}] Could not retrieve 'calculator' tool from registry. "
                               "This agent might not function as expected without it.")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to retrieve 'calculator' tool during setup: {e}")
            self.calculator_tool = None # Ensure it's None on failure

    async def run(self, event_bus: Any, state_manager: Any) -> None:
        """
        Executes the agent's main logic.
        It uses the 'calculator' tool to perform an operation defined in its configuration,
        updates its internal state, and potentially the global state, and logs the outcome.

        Args:
            event_bus (Any): The central message passing system for inter-agent communication.
                             (Type hint `Any` used to avoid circular/missing imports for this file).
            state_manager (Any): Manages the canonical state of all agents and the system.
                                 (Type hint `Any` used to avoid circular/missing imports for this file).
        """
        logger.info(f"[{self.agent_id}] Starting run cycle.")

        if not self.calculator_tool:
            logger.error(f"[{self.agent_id}] Cannot perform calculation: 'calculator' tool is not available. Exiting run.")
            # In a full system, an error message would be published to the event bus using a defined schema.
            # Example (assuming AgentMessage and MessageType are imported from protocols.message_schemas):
            # await event_bus.publish(
            #     AgentMessage(
            #         sender_id=self.agent_id,
            #         receiver_id="orchestrator", # Or a meta-agent
            #         message_type=MessageType.ERROR,
            #         payload={"error": "Required tool 'calculator' not found."}
            #     )
            # )
            # For this example, we'll send a simple dict as a placeholder message.
            await event_bus.publish(
                {"sender_id": self.agent_id, "type": "ERROR", "payload": "Required tool 'calculator' not found."}
            )
            return

        # Extract operation parameters from the agent's configuration
        operation = self.config.get("operation", "add") # Default operation
        num1 = self.config.get("num1", 10) # Default operand 1
        num2 = self.config.get("num2", 5) # Default operand 2

        logger.info(f"[{self.agent_id}] Attempting to use 'calculator' tool for '{operation}' with {num1} and {num2}.")

        try:
            # Execute the tool. We assume the 'calculator' tool is an awaitable callable
            # that takes the operation string and two numbers as arguments.
            result = await self.calculator_tool(operation, num1, num2)

            logger.info(f"[{self.agent_id}] Calculation successful: {num1} {operation} {num2} = {result}")

            # Update the agent's internal state with the latest calculation result
            self.state["last_calculation"] = {
                "operation": operation,
                "operands": [num1, num2],
                "result": result,
                "timestamp": state_manager.get_current_time() if hasattr(state_manager, 'get_current_time') else None
            }
            logger.debug(f"[{self.agent_id}] Agent internal state updated: {self.state}")

            # Also update the global state managed by the StateManager
            await state_manager.update_agent_state(
                self.agent_id,
                {"status": "completed_calculation", "last_result": result}
            )

            # Publish a message to the event bus indicating task completion and result.
            # This allows other agents or the orchestrator to react to this outcome.
            # Example (assuming AgentMessage and MessageType are imported from protocols.message_schemas):
            # await event_bus.publish(
            #     AgentMessage(
            #         sender_id=self.agent_id,
            #         receiver_id="orchestrator", # Or a specific listening agent
            #         message_type=MessageType.INFO,
            #         payload={
            #             "task": "calculation_performed",
            #             "operation": operation,
            #             "operands": [num1, num2],
            #             "result": result
            #         }
            #     )
            # )
            # For this example, we'll send a simple dict as a placeholder message.
            await event_bus.publish(
                {"sender_id": self.agent_id, "type": "INFO", "payload": {"task": "calculation_performed", "result": result}}
            )
            logger.info(f"[{self.agent_id}] Published calculation result to event bus: {result}")

        except KeyError:
            # Handle cases where the tool might not support the given operation
            logger.error(f"[{self.agent_id}] Invalid operation '{operation}' specified for calculator tool.")
            await event_bus.publish(
                # Placeholder for actual message schema
                {"sender_id": self.agent_id, "type": "ERROR", "payload": f"Invalid operation: '{operation}' for calculator tool."}
            )
        except Exception as e:
            # Catch any other exceptions during tool execution
            logger.error(f"[{self.agent_id}] Failed to execute calculation with tool for {operation} {num1}, {num2}: {e}")
            # Publish an error message to the event bus
            # Example:
            # await event_bus.publish(
            #     AgentMessage(
            #         sender_id=self.agent_id,
            #         receiver_id="orchestrator",
            #         message_type=MessageType.ERROR,
            #         payload={"task": "calculation_failed", "details": str(e)}
            #     )
            # )
            await event_bus.publish(
                # Placeholder for actual message schema
                {"sender_id": self.agent_id, "type": "ERROR", "payload": f"Calculation failed: {e}"}
            )

    async def teardown(self) -> None:
        """
        Performs asynchronous cleanup for the agent before it is terminated.
        """
        logger.info(f"[{self.agent_id}] Starting teardown phase.")
        # Any specific cleanup, like closing connections or releasing resources, would go here.
        # For this example, there are no specific resources to release.
        logger.info(f"[{self.agent_id}] Teardown complete.")