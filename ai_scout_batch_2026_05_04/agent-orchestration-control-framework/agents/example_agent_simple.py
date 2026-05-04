import asyncio
import logging
from typing import Dict, Any

from agents.base_agent import BaseAgent
from protocols.message_schemas import AgentID, AgentMessage
from core.event_bus import EventBus  # For type hinting the constructor

# It's good practice for each module to have its own logger, configured centrally
logger = logging.getLogger(__name__)

class ExampleAgentSimple(BaseAgent):
    """
    A simple example agent that demonstrates basic agent lifecycle,
    message sending, and message reception.

    It is configured to send 'ping' messages to a target agent periodically
    and to respond with 'pong' messages if it receives a 'ping_request'.
    This agent can act as both an initiator and a responder.
    """

    def __init__(self, agent_id: AgentID, config: Dict[str, Any], event_bus: EventBus):
        """
        Initializes the ExampleAgentSimple.

        Args:
            agent_id: The unique identifier for this agent.
            config: A dictionary containing agent-specific configuration.
                    Expected keys:
                    - "target_agent_id": The ID of the agent to send pings to.
                    - "ping_interval_seconds": How often to send pings (if initiating).
                    - "initiate_pings": Boolean, whether this agent should start sending pings.
            event_bus: The central EventBus instance for inter-agent communication.
        """
        super().__init__(agent_id, config, event_bus)
        
        # Validate and set configuration parameters
        try:
            self.target_agent_id: AgentID = AgentID(config.get("target_agent_id", "example_agent_responder"))
            self.ping_interval_seconds: int = int(config.get("ping_interval_seconds", 5))
            if self.ping_interval_seconds <= 0:
                raise ValueError("ping_interval_seconds must be a positive integer.")
            self.initiate_pings: bool = bool(config.get("initiate_pings", True))
        except (ValueError, TypeError) as e:
            self.logger.error(f"Agent '{self.agent_id}' failed to parse configuration: {e}")
            # Set defaults or raise to prevent agent from starting with bad config
            self.target_agent_id = AgentID("default_responder")
            self.ping_interval_seconds = 5
            self.initiate_pings = False # Disable initiation if config is bad
            self.logger.warning(f"Agent '{self.agent_id}' operating with default/fallback configuration due to error.")

        self.ping_count: int = 0
        self.logger.info(
            f"ExampleAgentSimple '{self.agent_id}' initialized. "
            f"Target for pings: '{self.target_agent_id}'. "
            f"Ping interval: {self.ping_interval_seconds}s. "
            f"Initiating pings: {self.initiate_pings}."
        )

    async def setup(self):
        """
        Agent setup phase.
        Registers handlers for messages this agent is interested in receiving.
        This method is called once when the agent is started.
        """
        self.logger.info(f"Agent '{self.agent_id}' performing setup.")
        
        # Register handlers for specific message types the agent expects
        self._register_message_handler("pong", self._handle_pong_message)
        self._register_message_handler("ping_request", self._handle_ping_request_message)
        
        # Simulate some asynchronous setup work, e.g., connecting to a service
        await asyncio.sleep(0.1) 
        self.logger.info(f"Agent '{self.agent_id}' setup complete.")

    async def run(self):
        """
        Main execution loop for the agent.
        If configured to initiate pings, this agent sends 'ping_request' messages
        periodically. Otherwise, it simply remains active to respond to messages.
        This method is called after setup and runs as long as the agent is active.
        """
        self.logger.info(f"Agent '{self.agent_id}' starting its main run loop.")
        try:
            if self.initiate_pings:
                while self.is_running():
                    await self._send_ping_message()
                    await asyncio.sleep(self.ping_interval_seconds)
            else:
                self.logger.info(
                    f"Agent '{self.agent_id}' is configured not to initiate pings. "
                    "Waiting passively for incoming 'ping_request' messages."
                )
                # Keep the agent alive to receive and process messages via its handlers
                while self.is_running():
                    await asyncio.sleep(1) # Sleep to prevent busy-waiting
        except asyncio.CancelledError:
            self.logger.info(f"Agent '{self.agent_id}' run loop was cancelled as part of shutdown.")
        except Exception as e:
            # Use logger.exception to log the traceback automatically
            self.logger.exception(f"Agent '{self.agent_id}' encountered an unhandled error in run loop.")
        finally:
            self.logger.info(f"Agent '{self.agent_id}' main run loop terminated.")

    async def teardown(self):
        """
        Agent teardown phase.
        Performs any necessary cleanup before the agent is stopped.
        This method is called once when the agent is being stopped.
        """
        self.logger.info(f"Agent '{self.agent_id}' performing teardown.")
        # In this simple example, there are no specific external resources (like database
        # connections or open files) to clean up.
        # Simulate some asynchronous cleanup work.
        await asyncio.sleep(0.1)
        self.logger.info(f"Agent '{self.agent_id}' teardown complete.")

    async def _send_ping_message(self):
        """
        Constructs and sends a 'ping_request' message to the configured target agent.
        Includes a sequence number and current timestamp for tracking.
        """
        self.ping_count += 1
        message_content = {
            "sequence": self.ping_count,
            "timestamp": asyncio.get_event_loop().time(),
            "data": f"Hello from {self.agent_id}! This is ping number {self.ping_count}.",
        }
        self.logger.debug(
            f"Agent '{self.agent_id}' preparing to send 'ping_request' "
            f"#{self.ping_count} to '{self.target_agent_id}'."
        )
        try:
            await self._send_message(
                recipient_id=self.target_agent_id,
                message_type="ping_request",
                content=message_content
            )
            self.logger.info(
                f"Agent '{self.agent_id}' successfully sent 'ping_request' "
                f"#{self.ping_count} to '{self.target_agent_id}'."
            )
        except Exception as e:
            self.logger.error(
                f"Agent '{self.agent_id}' failed to send 'ping_request' to '{self.target_agent_id}': {e}",
                exc_info=True
            )

    async def _handle_pong_message(self, message: AgentMessage):
        """
        Handles incoming 'pong' messages from other agents.
        Logs the details of the received pong message.
        """
        sender_id = message.sender_id
        sequence = message.content.get("sequence", "N/A")
        reply_data = message.content.get("data", "No data provided in pong.")
        
        self.logger.info(
            f"Agent '{self.agent_id}' received 'pong' (Seq: {sequence}) "
            f"from '{sender_id}': '{reply_data}'"
        )
        # Here, the agent could update its internal state, log metrics,
        # or trigger further actions based on the received response.

    async def _handle_ping_request_message(self, message: AgentMessage):
        """
        Handles incoming 'ping_request' messages and sends a 'pong' in response.
        This allows this agent to act as a responder to other agents' pings,
        regardless of its 'initiate_pings' configuration.
        """
        sender_id = message.sender_id
        sequence = message.content.get("sequence", "N/A")
        request_data = message.content.get("data", "No data provided in ping_request.")
        
        self.logger.info(
            f"Agent '{self.agent_id}' received 'ping_request' (Seq: {sequence}) "
            f"from '{sender_id}': '{request_data}'. Preparing to send 'pong' response."
        )
        
        response_content = {
            "sequence": sequence, # Echo back the sequence number for correlation
            "timestamp": asyncio.get_event_loop().time(),
            "data": f"Pong from {self.agent_id} to {sender_id}. Acknowledging your ping #{sequence}.",
        }
        
        try:
            await self._send_message(
                recipient_id=sender_id,
                message_type="pong",
                content=response_content
            )
            self.logger.info(
                f"Agent '{self.agent_id}' successfully sent 'pong' back to '{sender_id}' "
                f"for ping #{sequence}."
            )
        except Exception as e:
            self.logger.error(
                f"Agent '{self.agent_id}' failed to send 'pong' to '{sender_id}' for ping #{sequence}: {e}",
                exc_info=True
            )