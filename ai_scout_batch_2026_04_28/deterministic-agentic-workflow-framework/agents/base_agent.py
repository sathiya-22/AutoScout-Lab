```python
import abc
from typing import Any, Dict, Optional

# Assume these types are defined in other framework files as per the architecture notes.
# An ImportError will occur if these modules/classes do not exist, which is expected
# for a framework where components depend on each other.
from framework.workflow_manager import WorkflowManager
from utils.llm_connector import LLMConnector
from framework.action_schemas import AgentInput, AgentOutput # These should be Pydantic models
from framework.exceptions import AgentExecutionError


class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the Deterministic Agentic Workflow Framework.

    Enforces a standardized interface for agents, ensuring their actions and their expected outcomes
    are registered with the WorkflowManager and that they interact through the LLMConnector.
    """

    def __init__(
        self,
        name: str,
        workflow_manager: WorkflowManager,
        llm_connector: LLMConnector
    ):
        """
        Initializes the base agent.

        Args:
            name (str): A unique, descriptive name for the agent.
            workflow_manager (WorkflowManager): The workflow manager instance responsible for
                                                state management and action registration.
            llm_connector (LLMConnector): The LLM connector instance for making standardized LLM calls.

        Raises:
            ValueError: If the agent name is not a non-empty string.
            TypeError: If workflow_manager or llm_connector are not instances of their respective types.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent name must be a non-empty string.")
        if not isinstance(workflow_manager, WorkflowManager):
            raise TypeError(f"workflow_manager must be an instance of WorkflowManager, got {type(workflow_manager).__name__}.")
        if not isinstance(llm_connector, LLMConnector):
            raise TypeError(f"llm_connector must be an instance of LLMConnector, got {type(llm_connector).__name__}.")

        self._name = name.strip()
        self._workflow_manager = workflow_manager
        self._llm_connector = llm_connector

    @property
    def name(self) -> str:
        """Returns the name of the agent."""
        return self._name

    @abc.abstractmethod
    async def execute(self, task_input: AgentInput, **kwargs) -> AgentOutput:
        """
        Abstract method to be implemented by concrete agents.
        This method defines the agent's core logic and task execution.

        Args:
            task_input (AgentInput): The structured input data for the agent's task.
                                     This should conform to a Pydantic model from action_schemas.py.
            **kwargs: Additional keyword arguments for execution, specific to the agent's task.
                      These might include context, retry attempts, specific configuration, etc.

        Returns:
            AgentOutput: The structured output of the agent's execution.
                         This should conform to a Pydantic model from action_schemas.py.

        Raises:
            NotImplementedError: If the concrete agent does not implement this method.
            AgentExecutionError: For errors encountered during the agent's execution logic.
        """
        raise NotImplementedError("Concrete agents must implement the 'execute' method.")

    async def _call_llm(self, model: str, messages: Any, **kwargs) -> Any:
        """
        Helper method for agents to interact with the LLMConnector.
        This method provides a standardized and robust way to make LLM calls within the framework,
        potentially incorporating retry logic, rate limiting, and structured logging defined in LLMConnector.

        Args:
            model (str): The name of the LLM model to use (e.g., "gpt-4", "claude-3-opus", "local-model").
            messages (Any): The messages payload for the LLM chat completion,
                            typically a list of dictionaries conforming to the LLM provider's API specification.
            **kwargs: Additional keyword arguments to pass directly to the LLM connector's chat_completion method
                      (e.g., `temperature`, `max_tokens`, `response_format`).

        Returns:
            Any: The raw response object from the LLM. It is the responsibility of the concrete agent
                 to parse and validate this response into a structured AgentOutput.

        Raises:
            AgentExecutionError: If the LLM call fails (e.g., API error, network issue, parsing failure
                                 within the connector).
        """
        try:
            llm_response = await self._llm_connector.chat_completion(model=model, messages=messages, **kwargs)
            return llm_response
        except Exception as e:
            # Wrap the exception in a framework-specific error for consistent handling and traceability.
            raise AgentExecutionError(
                f"Agent '{self.name}' failed to call LLM with model '{model}': {e}"
            ) from e

    async def _register_action_and_output(
        self,
        action_type: str,
        inputs: AgentInput,
        outputs: AgentOutput,
        description: Optional[str] = None
    ) -> None:
        """
        Helper method for agents to formally register their executed actions and their outputs
        with the WorkflowManager. This is a critical step for state tracking, semantic comparison,
        checkpointing, and potential reconciliation processes.

        Args:
            action_type (str): A string representing the semantic type of action performed by the agent
                               (e.g., "data_processing", "summary_generation", "decision_making").
            inputs (AgentInput): The structured input (Pydantic model) that initiated this specific action.
            outputs (AgentOutput): The structured output (Pydantic model) produced by this action.
            description (Optional[str]): A human-readable description or summary of the action taken,
                                         useful for logging and debugging.

        Raises:
            AgentExecutionError: If the registration process with the WorkflowManager fails.
        """
        try:
            await self._workflow_manager.register_agent_action(
                agent_name=self.name,
                action_type=action_type,
                inputs=inputs,
                outputs=outputs,
                description=description
            )
        except Exception as e:
            # Wrap the exception for consistent error handling within the framework.
            raise AgentExecutionError(
                f"Agent '{self.name}' failed to register action '{action_type}' with workflow manager: {e}"
            ) from e
```