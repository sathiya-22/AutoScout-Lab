import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from llm_proxies.base_proxy import BaseLLMProxy

class BaseAgent(ABC):
    """
    Abstract Base Class for all agents in the multi-agent LLM system.

    Provides common functionality for agents, including interaction with LLMs
    via a specified LLM proxy and basic logging. Concrete agent classes must
    implement the `execute` method to define their specific task logic.
    """

    def __init__(self, name: str, llm_proxy: BaseLLMProxy):
        """
        Initializes the BaseAgent.

        Args:
            name (str): The unique name of the agent. Must be a non-empty string.
            llm_proxy (BaseLLMProxy): The LLM proxy instance this agent will use
                                       for all LLM interactions.

        Raises:
            ValueError: If the agent name is invalid.
            TypeError: If llm_proxy is not an instance of BaseLLMProxy.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent name must be a non-empty string.")
        if not isinstance(llm_proxy, BaseLLMProxy):
            raise TypeError("llm_proxy must be an instance of BaseLLMProxy.")

        self.name = name.strip()
        self.llm_proxy = llm_proxy
        self.logger = logging.getLogger(f"Agent.{self.name}")
        self.logger.info(f"Agent '{self.name}' initialized with LLM proxy: {type(llm_proxy).__name__}")

    async def invoke_llm(self, prompt: str, **kwargs) -> Any:
        """
        Invokes the underlying LLM via the configured proxy.

        This method abstracts the direct interaction with the LLM, routing
        all calls through the agent's designated LLM proxy, which might
        implement features like seeding, caching, or structured output.

        Args:
            prompt (str): The prompt string to send to the LLM.
            **kwargs: Additional keyword arguments to pass to the LLM proxy's invoke method.
                      These might include parameters like temperature, max_tokens, etc.

        Returns:
            Any: The raw or structured response from the LLM.

        Raises:
            RuntimeError: If the LLM invocation fails due to an underlying proxy error.
        """
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")
        
        # Log the first 100 characters of the prompt for brevity in logs
        self.logger.debug(f"Agent '{self.name}' invoking LLM with prompt (first 100 chars): '{prompt[:100]}...'")
        try:
            response = await self.llm_proxy.invoke(prompt, **kwargs)
            self.logger.debug(f"Agent '{self.name}' successfully received LLM response.")
            return response
        except Exception as e:
            self.logger.error(
                f"Agent '{self.name}' failed to invoke LLM with proxy {type(self.llm_proxy).__name__}: {e}",
                exc_info=True
            )
            raise RuntimeError(f"LLM invocation failed for agent '{self.name}': {e}") from e

    @abstractmethod
    async def execute(self, task_input: Any, context: Dict[str, Any] = None) -> Any:
        """
        Abstract method for the agent to execute its primary task.

        Concrete agent classes must implement this method to define their
        specific operational logic, which may involve multiple LLM invocations,
        data processing, or interactions with other systems.

        Args:
            task_input (Any): The primary input relevant to the agent's current task.
                              The type of this input depends on the specific agent's role.
            context (Dict[str, Any], optional): A dictionary providing additional
                                                context or shared state for the agent's execution.
                                                This can be used to pass information from an
                                                orchestrator or other agents. Defaults to None.

        Returns:
            Any: The result of the agent's task execution. The type of this output
                 depends on the specific agent's task and desired output format.
        """
        pass<ctrl63>