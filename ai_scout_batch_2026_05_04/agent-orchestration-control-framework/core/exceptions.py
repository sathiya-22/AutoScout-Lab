class OrchestrationFrameworkError(Exception):
    """Base exception for all errors within the Agent Orchestration Control Framework."""
    pass


class ConfigurationError(OrchestrationFrameworkError):
    """Exception raised for issues related to configuration loading or validation."""
    def __init__(self, message: str, config_path: str = None):
        super().__init__(f"Configuration Error: {message}" + (f" (Path: {config_path})" if config_path else ""))
        self.config_path = config_path


class AgentError(OrchestrationFrameworkError):
    """Base exception for errors related to agents."""
    pass


class AgentNotFound(AgentError):
    """Exception raised when an agent with a given ID or name is not found."""
    def __init__(self, agent_id: str):
        super().__init__(f"Agent with ID '{agent_id}' not found.")
        self.agent_id = agent_id


class InvalidAgentStateError(AgentError):
    """Exception raised when an agent is in an unexpected or invalid state."""
    def __init__(self, agent_id: str, current_state: str, expected_states: list = None):
        if expected_states:
            super().__init__(f"Agent '{agent_id}' is in state '{current_state}', but expected one of {expected_states}.")
        else:
            super().__init__(f"Agent '{agent_id}' is in an invalid state: '{current_state}'.")
        self.agent_id = agent_id
        self.current_state = current_state
        self.expected_states = expected_states


class StateManagerError(OrchestrationFrameworkError):
    """Base exception for errors related to the StateManager."""
    pass


class InvalidStateTransitionError(StateManagerError):
    """Exception raised when an attempted state transition is invalid."""
    def __init__(self, entity_id: str, current_state: str, target_state: str, reason: str = None):
        message = f"Invalid state transition for '{entity_id}' from '{current_state}' to '{target_state}'."
        if reason:
            message += f" Reason: {reason}"
        super().__init__(message)
        self.entity_id = entity_id
        self.current_state = current_state
        self.target_state = target_state
        self.reason = reason


class OrchestratorError(OrchestrationFrameworkError):
    """Base exception for errors within the Orchestrator."""
    pass


class SchedulingError(OrchestratorError):
    """Exception raised when an agent action cannot be scheduled or executed due to scheduling conflicts or logic."""
    def __init__(self, agent_id: str, action: str, reason: str):
        super().__init__(f"Scheduling error for agent '{agent_id}' action '{action}': {reason}")
        self.agent_id = agent_id
        self.action = action
        self.reason = reason


class CommunicationError(OrchestrationFrameworkError):
    """Base exception for errors related to inter-agent communication."""
    pass


class MessageSchemaError(CommunicationError):
    """Exception raised when a message does not conform to its defined schema."""
    def __init__(self, message_type: str, details: str):
        super().__init__(f"Message of type '{message_type}' failed schema validation: {details}")
        self.message_type = message_type
        self.details = details


class ProtocolViolationError(CommunicationError):
    """Exception raised when an agent violates a defined communication protocol."""
    def __init__(self, agent_id: str, protocol_name: str, violation_details: str):
        super().__init__(f"Agent '{agent_id}' violated protocol '{protocol_name}': {violation_details}")
        self.agent_id = agent_id
        self.protocol_name = protocol_name
        self.violation_details = violation_details


class ToolError(OrchestrationFrameworkError):
    """Base exception for errors related to tool management or execution."""
    pass


class ToolNotFound(ToolError):
    """Exception raised when a requested tool is not found in the registry."""
    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' not found in the registry.")
        self.tool_name = tool_name


class ToolExecutionError(ToolError):
    """Exception raised when a tool execution fails."""
    def __init__(self, tool_name: str, agent_id: str, original_exception: Exception):
        super().__init__(f"Tool '{tool_name}' execution failed for agent '{agent_id}': {original_exception}")
        self.tool_name = tool_name
        self.agent_id = agent_id
        self.original_exception = original_exception


class SpecificationError(OrchestrationFrameworkError):
    """Base exception for errors related to DSL parsing, models, or constraint enforcement."""
    pass


class DSLParsingError(SpecificationError):
    """Exception raised when there's an error parsing the Domain Specific Language."""
    def __init__(self, filepath: str, details: str, line: int = None, column: int = None):
        location = f" at line {line}, column {column}" if line is not None and column is not None else ""
        super().__init__(f"Error parsing DSL file '{filepath}'{location}: {details}")
        self.filepath = filepath
        self.line = line
        self.column = column
        self.details = details


class ConstraintViolationError(SpecificationError):
    """Exception raised when a defined constraint is violated during runtime."""
    def __init__(self, constraint_id: str, entity_id: str, details: str):
        super().__init__(f"Constraint '{constraint_id}' violated by '{entity_id}': {details}")
        self.constraint_id = constraint_id
        self.entity_id = entity_id
        self.details = details


class PolicyViolationError(OrchestrationFrameworkError):
    """Exception raised when an operational policy defined in the policy engine is violated."""
    def __init__(self, policy_id: str, agent_id: str, details: str):
        super().__init__(f"Policy '{policy_id}' violated by agent '{agent_id}': {details}")
        self.policy_id = policy_id
        self.agent_id = agent_id
        self.details = details


class ControlPlaneError(OrchestrationFrameworkError):
    """Base exception for errors originating from the control plane operations."""
    def __init__(self, message: str):
        super().__init__(f"Control Plane Error: {message}")


class DebuggingError(OrchestrationFrameworkError):
    """Base exception for errors related to debugging and tracing functionalities."""
    def __init__(self, message: str):
        super().__init__(f"Debugging Error: {message}")