import abc
from typing import Any, Dict, Optional, Type
from abc import ABC, abstractmethod
import asyncio # Required for async operations

# --- Placeholder for LLM client integration (as described in architecture point 9) ---
# In a full system, this would be imported from `llm_integrations/llm_client.py`.
# For the purpose of defining `base_agent.py` in isolation, we'll define a simple
# abstract base and a mock concrete implementation here.
class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    Concrete implementations will interact with specific LLM providers (e.g., OpenAI, Anthropic).
    """
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self._kwargs = kwargs # Store additional parameters for specific LLM config

    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Abstract method to generate a response from the LLM.

        Args:
            prompt (str): The input prompt for the LLM.
            **kwargs: Additional parameters specific to the LLM call (e.g., temperature, max_tokens).

        Returns:
            str: The raw string response from the LLM.
        """
        pass

# A basic concrete LLMClient for internal use in this file, primarily for testing
# or when no external LLMClient is explicitly passed to BaseAgent.
# In a real setup, concrete LLMClient implementations (e.g., OpenAIClient, AnthropicClient)
# would reside in `llm_integrations/llm_client.py`.
class MockLLMClient(LLMClient):
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Simulates an LLM response. This is a mock implementation.
        """
        # Simulate network latency
        await asyncio.sleep(0.1)
        response_template = f"Mock LLM response from {self.model_name} for prompt (first 50 chars): '{prompt[:50].replace('\\n', ' ')}...'. Parameters: {kwargs}"
        return response_template

# --- Placeholder for Configuration (as described in architecture point 10) ---
# In a full system, this would be imported from `config/settings.py`.
class Config:
    """
    A placeholder for global configuration settings.
    These values would typically be loaded from a config file (e.g., YAML, .env)
    and managed by a dedicated configuration module.
    """
    LLM_MODEL: str = "gpt-4-turbo-preview"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 4096
    AGENT_LOGGING_ENABLED: bool = True
    # Add other common settings needed by agents

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the hierarchical system.
    Provides common functionalities like LLM interaction, configuration management,
    and a structured approach to task processing.
    """

    def __init__(self, agent_name: str, llm_client: Optional[LLMClient] = None, **kwargs):
        """
        Initializes the BaseAgent.

        Args:
            agent_name (str): The unique name of the agent.
            llm_client (Optional[LLMClient]): An instance of an LLM client.
                                            If None, a default one (MockLLMClient for this prototype)
                                            will be initialized using configuration settings.
            **kwargs: Additional configuration parameters specific to the agent,
                      which can override default or global settings.
        """
        if not agent_name:
            raise ValueError("Agent name cannot be empty.")

        self._agent_name = agent_name
        self._config = self._load_agent_config(**kwargs)

        # Initialize LLM client. If provided, use it; otherwise, initialize a default.
        if llm_client:
            if not isinstance(llm_client, LLMClient):
                raise TypeError("Provided llm_client must be an instance of LLMClient.")
            self._llm_client = llm_client
        else:
            self._llm_client = self._initialize_default_llm_client()

        self.log(f"Agent '{self._agent_name}' initialized with LLM model: {self._llm_client.model_name}")

    def _load_agent_config(self, **kwargs) -> Dict[str, Any]:
        """
        Loads configuration for the agent, potentially overriding global settings
        from the placeholder `Config` class.
        """
        agent_config = {
            "llm_model": Config.LLM_MODEL,
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
            "logging_enabled": Config.AGENT_LOGGING_ENABLED,
            # Add other common config items that might be globally defined
        }
        # Override with agent-specific kwargs
        agent_config.update(kwargs)
        return agent_config

    def _initialize_default_llm_client(self) -> LLMClient:
        """
        Initializes a default LLM client using configured settings.
        For this prototype, it uses `MockLLMClient`. In a full system, this would
        use a factory from `llm_integrations/llm_client.py` to get a concrete LLM client.
        """
        try:
            return MockLLMClient(
                model_name=self._config.get("llm_model", Config.LLM_MODEL),
                temperature=self._config.get("temperature", Config.TEMPERATURE),
                max_tokens=self._config.get("max_tokens", Config.MAX_TOKENS)
            )
        except Exception as e:
            self.log(f"Error initializing default LLM client: {e}", level="ERROR")
            raise # Re-raise to indicate a critical setup failure

    @property
    def name(self) -> str:
        """Returns the name of the agent."""
        return self._agent_name

    @property
    def llm_client(self) -> LLMClient:
        """Returns the LLM client instance used by this agent."""
        return self._llm_client

    @property
    def config(self) -> Dict[str, Any]:
        """Returns the agent's configuration."""
        return self._config

    def log(self, message: str, level: str = "INFO"):
        """
        Basic logging mechanism for the agent.
        Should be replaced by a proper logging utility in a production system.
        """
        if self._config.get("logging_enabled", True):
            print(f"[{level}] [{self.name}] {message}")

    async def _call_llm(self, prompt: str, **kwargs) -> str:
        """
        Helper method to call the LLM and encapsulate common error handling.
        This method uses the agent's configured LLM client.

        Args:
            prompt (str): The prompt string to send to the LLM.
            **kwargs: Overrides for LLM call parameters (e.g., temperature, max_tokens)
                      for this specific call. Defaults come from agent's config.

        Returns:
            str: The raw string response from the LLM.

        Raises:
            Exception: If the LLM call fails for any reason.
        """
        # Merge call-specific kwargs with agent's default LLM parameters
        llm_params = {
            "temperature": self._config.get("temperature", Config.TEMPERATURE),
            "max_tokens": self._config.get("max_tokens", Config.MAX_TOKENS),
            # Add other common LLM parameters here
        }
        llm_params.update(kwargs)

        try:
            self.log(f"Calling LLM ({self._llm_client.model_name}) with prompt (first 100 chars): '{prompt[:100].replace('\\n', ' ')}...'")
            response = await self._llm_client.generate_response(prompt, **llm_params)
            self.log(f"Received LLM response (first 100 chars): '{response[:100].replace('\\n', ' ')}...'")
            return response
        except Exception as e:
            self.log(f"Error calling LLM for agent '{self.name}': {type(e).__name__} - {e}", level="ERROR")
            # Implement retry logic or specific error handling here if necessary
            raise # Re-raise the exception after logging for upstream handling

    @abstractmethod
    async def process_task(self, task_input: Any, **kwargs) -> Any:
        """
        Abstract method to be implemented by concrete agents for processing a task.
        This is the core logic method for each specialized agent.

        Args:
            task_input (Any): The input data or task description for the agent.
            **kwargs: Additional parameters specific to the task processing.

        Returns:
            Any: The result of the task processing.
        """
        raise NotImplementedError("Each concrete agent must implement its own process_task method.")