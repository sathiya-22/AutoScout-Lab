```python
import abc
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

# Assuming these imports will be available from other modules
# Adjust import paths if necessary based on project root.
from src.utils.llm_interface import LLMInterface
from src.prompting.prompt_manager import PromptManager


class BaseAgent(ABC):
    """
    Abstract base class for all LLM agents in the system.
    Provides common functionality for interacting with LLMs and managing prompts.
    """

    def __init__(self,
                 llm_interface: LLMInterface,
                 prompt_manager: PromptManager,
                 agent_name: str,
                 model_name: str = "gpt-4-turbo",
                 temperature: float = 0.7,
                 max_tokens: int = 1024):
        """
        Initializes the BaseAgent.

        Args:
            llm_interface: An instance of LLMInterface for interacting with LLM providers.
            prompt_manager: An instance of PromptManager for loading and managing prompts.
            agent_name: A unique name for the agent (e.g., "PrimaryAgent", "ValidationAgent").
            model_name: The specific LLM model to use (e.g., "gpt-4-turbo").
            temperature: The sampling temperature for the LLM.
            max_tokens: The maximum number of tokens to generate.
        """
        if not isinstance(llm_interface, LLMInterface):
            raise TypeError("llm_interface must be an instance of LLMInterface.")
        if not isinstance(prompt_manager, PromptManager):
            raise TypeError("prompt_manager must be an instance of PromptManager.")
        if not isinstance(agent_name, str) or not agent_name:
            raise ValueError("agent_name must be a non-empty string.")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be a float between 0.0 and 2.0.")
        if not isinstance(max_tokens, int) or not (max_tokens > 0):
            raise ValueError("max_tokens must be a positive integer.")

        self._llm_interface = llm_interface
        self._prompt_manager = prompt_manager
        self._agent_name = agent_name
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def agent_name(self) -> str:
        """Returns the name of the agent."""
        return self._agent_name

    @property
    def model_name(self) -> str:
        """Returns the LLM model name used by the agent."""
        return self._model_name

    @property
    def temperature(self) -> float:
        """Returns the temperature setting for the LLM."""
        return self._temperature

    @property
    def max_tokens(self) -> int:
        """Returns the maximum number of tokens for LLM generation."""
        return self._max_tokens

    @abstractmethod
    async def process_request(self, *args, **kwargs) -> Any:
        """
        Abstract method to be implemented by concrete agent classes.
        This method defines how the agent processes a request and generates a response
        using the LLM.

        Args:
            *args: Variable length argument list specific to the concrete agent's processing.
            **kwargs: Arbitrary keyword arguments specific to the concrete agent's processing.

        Returns:
            Any: The agent's processed output, which could be a tool call, a textual response, etc.
        """
        pass

    async def _call_llm(self,
                        messages: List[Dict[str, str]],
                        tools: Optional[List[Dict[str, Any]]] = None,
                        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                        **kwargs) -> Any:
        """
        Internal method to interact with the LLM via the LLMInterface.

        Args:
            messages: A list of message dictionaries for the LLM conversation (e.g., [{"role": "user", "content": "..."}]).
            tools: Optional list of tool definitions for the LLM, typically in OpenAI function calling format.
            tool_choice: Optional parameter for controlling tool usage (e.g., "auto", "none", {"type": "function", "function": {"name": "my_tool"}}).
            **kwargs: Additional parameters to pass to the underlying LLM chat completion method.

        Returns:
            Any: The raw response object from the LLM interface. This typically needs further parsing
                 by the concrete agent to extract content or tool calls.

        Raises:
            Exception: Re-raises any exceptions encountered during the LLM call after logging.
        """
        try:
            self._log_agent_activity(f"Calling LLM with model '{self._model_name}' and {len(messages)} messages.", level="debug")
            response = await self._llm_interface.chat_completion(
                model=self._model_name,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                **kwargs
            )
            self._log_agent_activity(f"LLM call successful. Response type: {type(response)}", level="debug")
            return response
        except Exception as e:
            error_msg = f"Error calling LLM for agent '{self._agent_name}': {e}"
            self._log_agent_activity(error_msg, level="error")
            raise Exception(error_msg) from e

    def _get_prompt_messages(self, prompt_name: str, **kwargs) -> List[Dict[str, str]]:
        """
        Retrieves and formats a prompt from the PromptManager.

        Args:
            prompt_name: The name of the prompt template to retrieve (e.g., "primary_agent_system_prompt").
            **kwargs: Variables to format the prompt template.

        Returns:
            List[Dict[str, str]]: The formatted prompt messages ready for the LLM.

        Raises:
            Exception: Re-raises any exceptions encountered during prompt retrieval/formatting after logging.
        """
        try:
            self._log_agent_activity(f"Retrieving prompt '{prompt_name}'", level="debug")
            return self._prompt_manager.get_prompt_messages(prompt_name, **kwargs)
        except Exception as e:
            error_msg = f"Error retrieving prompt '{prompt_name}' for agent '{self._agent_name}': {e}"
            self._log_agent_activity(error_msg, level="error")
            raise Exception(error_msg) from e

    def _log_agent_activity(self, message: str, level: str = "info"):
        """
        Internal logging utility for agent activities.
        In a production system, this would integrate with a structured logging framework
        (e.g., Python's built-in `logging` module). For this prototype, a simple print
        statement with timestamp and agent context is used.

        Args:
            message: The log message.
            level: The logging level (e.g., "info", "debug", "warning", "error").
        """
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        print(f"[{timestamp}][{self._agent_name}][{level.upper()}]: {message}")
```