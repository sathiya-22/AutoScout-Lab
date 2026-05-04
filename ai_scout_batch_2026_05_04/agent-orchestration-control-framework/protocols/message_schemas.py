```python
from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, Any, List
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum

# --- Type Aliases ---

AgentID = str
"""Type alias for agent identifiers, typically a string or unique name."""

# --- Enums ---

class MessageType(str, Enum):
    """
    Defines the high-level categories of messages exchanged between agents
    or between agents and the orchestration system.
    """
    COMMAND = "command"           # An agent requesting another agent or system to perform an action.
    RESPONSE = "response"         # A reply to a previously issued command or query.
    EVENT = "event"               # A broadcast notification about a significant occurrence.
    ERROR = "error"               # Reporting of an error or exception.
    BROADCAST = "broadcast"       # A general message intended for all interested parties.
    STATE_UPDATE = "state_update" # Communication of changes to an agent's internal state or global system state.
    DEBUG_INFO = "debug_info"     # For sending debugging or tracing information (e.g., intermediate thoughts, internal state).

# --- Base Message Schemas ---

class BaseMessage(BaseModel):
    """
    Base schema for all inter-agent messages, containing common metadata.
    This schema provides the envelope for all communications within the framework.
    """
    sender_id: AgentID = Field(..., description="The ID of the agent or system component sending the message.")
    receiver_id: Optional[AgentID] = Field(None, description="The ID of the intended recipient agent or component. None for broadcast or system-wide messages.")
    message_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this specific message instance.")
    conversation_id: Optional[UUID] = Field(None, description="Identifier to group related messages into a logical conversation flow. Helps trace interaction sequences.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp when the message was created.")
    message_type: MessageType = Field(..., description="The high-level type of the message, used for deserialization and routing.")

    class Config:
        use_enum_values = True  # Pydantic will serialize Enum members to their string values
        extra = "forbid"        # Disallow extra fields not defined in the schema to ensure strict adherence.

# --- Specific Message Payloads ---

class CommandPayload(BaseModel):
    """
    Payload schema for a 'COMMAND' message.
    Used when one agent requests another agent or the system to perform an action,
    often invoking a registered tool or internal method.
    """
    tool_name: str = Field(..., description="The name of the tool, capability, or internal action to invoke.")
    method_name: Optional[str] = Field(None, description="The specific method or function within the tool/capability to call. If None, implies a default action for the tool.")
    args: Optional[List[Any]] = Field([], description="Positional arguments for the command method.")
    kwargs: Optional[Dict[str, Any]] = Field({}, description="Keyword arguments for the command method.")
    
    class Config:
        extra = "forbid"

class ResponsePayload(BaseModel):
    """
    Payload schema for a 'RESPONSE' message.
    Used to reply to a previously issued 'COMMAND' or 'QUERY', indicating success, failure, or result.
    """
    status: str = Field(..., description="The execution status of the command (e.g., 'success', 'failure', 'pending', 'cancelled').")
    result: Optional[Any] = Field(None, description="The result data if the command was successful. Can be any serializable Python object.")
    error_message: Optional[str] = Field(None, description="A human-readable error message if the command failed.")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional structured details about any error that occurred.")
    
    class Config:
        extra = "forbid"

class EventPayload(BaseModel):
    """
    Payload schema for an 'EVENT' message.
    Used for broadcasting notifications about significant occurrences within the system,
    allowing other agents or monitoring components to react.
    """
    event_name: str = Field(..., description="A unique name identifying the type of event (e.g., 'agent_initialized', 'tool_executed', 'state_changed', 'new_task_assigned').")
    event_data: Optional[Dict[str, Any]] = Field(None, description="Arbitrary data associated with the event. Should be JSON-serializable.")
    
    class Config:
        extra = "forbid"

class ErrorPayload(BaseModel):
    """
    Payload schema for an 'ERROR' message.
    Used to report critical errors or exceptions encountered by an agent or the system component.
    """
    error_code: str = Field(..., description="A standardized short code identifying the type of error (e.g., 'TOOL_EXEC_FAILED', 'INVALID_ARGUMENTS').")
    error_message: str = Field(..., description="A comprehensive, human-readable description of the error.")
    component: Optional[str] = Field(None, description="The component or agent where the error originated.")
    traceback: Optional[str] = Field(None, description="Full Python traceback of the error, if available, for in-depth debugging.")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context or parameters that were active at the time of the error.")
    
    class Config:
        extra = "forbid"

class StateUpdatePayload(BaseModel):
    """
    Payload schema for a 'STATE_UPDATE' message.
    Used to communicate changes to an agent's internal state or global system state,
    facilitating consistent state management and synchronization.
    """
    agent_id: Optional[AgentID] = Field(None, description="The ID of the agent whose state is being updated. If None, implies a global system state update.")
    # Provide either a partial update (diff) or a full snapshot.
    state_diff: Optional[Dict[str, Any]] = Field(None, description="A dictionary representing the incremental changes to the state. Keys are state paths, values are new values.")
    full_state_snapshot: Optional[Dict[str, Any]] = Field(None, description="A complete snapshot of the relevant state. Use sparingly for large states to avoid overhead.")
    
    class Config:
        extra = "forbid"

class DebugInfoPayload(BaseModel):
    """
    Payload schema for a 'DEBUG_INFO' message.
    Used for sending debugging or tracing information, such as LLM thought processes,
    internal state dumps, or tool outputs, to aid in observability and debugging.
    """
    debug_type: str = Field(..., description="Type of debug information (e.g., 'llm_thought', 'internal_state_snapshot', 'tool_input', 'tool_output').")
    data: Dict[str, Any] = Field(..., description="The actual debugging data. Should be structured for easy analysis.")

    class Config:
        extra = "forbid"

# --- Concrete Message Schemas (combining BaseMessage with specific Payloads) ---

class CommandMessage(BaseMessage):
    """
    Represents a request for an agent or the system to perform an action.
    """
    message_type: MessageType = Field(MessageType.COMMAND, const=True, description="The type of message (fixed to 'command').")
    payload: CommandPayload = Field(..., description="The command-specific payload.")

class ResponseMessage(BaseMessage):
    """
    Represents a response to a previously issued command.
    """
    message_type: MessageType = Field(MessageType.RESPONSE, const=True, description="The type of message (fixed to 'response').")
    payload: ResponsePayload = Field(..., description="The response-specific payload.")

class EventMessage(BaseMessage):
    """
    Represents an event or notification broadcast by an agent or the system.
    """
    message_type: MessageType = Field(MessageType.EVENT, const=True, description="The type of message (fixed to 'event').")
    payload: EventPayload = Field(..., description="The event-specific payload.")

class ErrorMessage(BaseMessage):
    """
    Represents an error encountered by an agent or the system.
    """
    message_type: MessageType = Field(MessageType.ERROR, const=True, description="The type of message (fixed to 'error').")
    payload: ErrorPayload = Field(..., description="The error-specific payload.")

class StateUpdateMessage(BaseMessage):
    """
    Represents an update to an agent's or the overall system's state.
    """
    message_type: MessageType = Field(MessageType.STATE_UPDATE, const=True, description="The type of message (fixed to 'state_update').")
    payload: StateUpdatePayload = Field(..., description="The state update-specific payload.")

class DebugInfoMessage(BaseMessage):
    """
    Represents debugging or tracing information from an agent or system component.
    """
    message_type: MessageType = Field(MessageType.DEBUG_INFO, const=True, description="The type of message (fixed to 'debug_info').")
    payload: DebugInfoPayload = Field(..., description="The debug information specific payload.")

# --- Union Type for Generic Message Handling ---

AgentCommunicationMessage = Union[
    CommandMessage,
    ResponseMessage,
    EventMessage,
    ErrorMessage,
    StateUpdateMessage,
    DebugInfoMessage,
]
"""
A union type representing any valid inter-agent communication message.
Useful for type hints and for dynamically parsing incoming messages based on their 'message_type'.
When parsing raw data, one typically first parses to a generic `BaseMessage` to extract `message_type`,
then re-parses the full data into the specific concrete message schema.
"""
```