```python
import os
import abc
import json
from typing import List, Dict, Any, Optional, Union, Type, Awaitable

# Mock or actual import for configuration.
# In a real project, this would typically be 'from config import settings'.
# For standalone execution and to satisfy the prompt, we'll create a mock.
class MockSettings:
    """Mock configuration settings to avoid circular dependencies or requiring a full config module for this file."""
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")

    LOCAL_MODEL_API_BASE = os.getenv("LOCAL_MODEL_API_BASE", "http://localhost:11434/v1") # Default Ollama API
    LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "llama3")

    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", 0.7))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_LLM_MAX_TOKENS", 4096))
    DEFAULT_TIMEOUT_SECONDS = int(os.getenv("DEFAULT_LLM_TIMEOUT_SECONDS", 60))

settings = MockSettings()

# --- Abstract Base Class for LLM Clients ---
class BaseLLMClient(abc.ABC):
    """
    Abstract base class for all LLM clients.
    Defines the interface for interacting with different LLM providers,
    ensuring a consistent API across various LLM integrations.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    @abc.abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request to the LLM.

        Args:
            messages: A list of message dictionaries following the OpenAI format
                      `[{"role": "user", "content": "..."}]`.
            tools: An optional list of tool definitions. These are expected to be in a
                   format similar to OpenAI's function calling specification, which
                   concrete clients will adapt if necessary for their respective APIs.
            tool_choice: Controls how the model calls tools. Can be "auto", "none",
                         or a specific tool call definition (e.g., {"type": "function", "function": {"name": "my_tool"}}).
                         The specific format may vary slightly between providers.
            temperature: Sampling temperature to use. Lower values make the output
                         more deterministic, higher values make it more random.
            max_tokens: The maximum number of tokens to generate in the completion.
            timeout: Request timeout in seconds.
            **kwargs: Additional provider-specific parameters to pass directly to the
                      underlying API call.

        Returns:
            A dictionary representing the LLM's response, normalized to include:
            - 'content': The main textual content of the assistant's reply.
            - 'tool_calls': A list of tool calls, each a dictionary with 'id', 'type', and 'function' keys.
            - 'finish_reason': The reason the model stopped generating tokens (e.g., "stop", "tool_calls", "length").
            - 'model': The model that generated the response.
            - 'usage': A dictionary containing token usage statistics.
            - 'id': A unique identifier for the response.
            - 'raw_response': The original, unprocessed response object or its JSON representation.
        Raises:
            ValueError: If an invalid request is made (e.g., bad parameters, authentication error).
            RuntimeError: For transient API errors, rate limits, or unexpected issues.
        """
        pass

# --- Concrete OpenAI Client ---
try:
    import openai
    from openai import OpenAI
    from openai import APIError, RateLimitError, AuthenticationError, BadRequestError
except ImportError:
    openai = None
    OpenAI = None
    APIError, RateLimitError, AuthenticationError, BadRequestError = None, None, None, None
    print("Warning: OpenAI library not found. OpenAI client will be unavailable.")

class OpenAIChatClient(BaseLLMClient):
    """
    LLM client for OpenAI models, handling chat completions and tool calls.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        if openai is None:
            raise RuntimeError("OpenAI library not installed. Cannot use OpenAIChatClient.")
        super().__init__(model_name, api_key)
        self.client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Sends a chat completion request to OpenAI."""
        if temperature is None:
            temperature = settings.DEFAULT_TEMPERATURE
        if max_tokens is None:
            max_tokens = settings.DEFAULT_MAX_TOKENS
        if timeout is None:
            timeout = settings.DEFAULT_TIMEOUT_SECONDS

        request_payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            **kwargs
        }

        if tools:
            # OpenAI expects 'tools' as a list of dictionaries with 'type' and 'function' keys.
            # We assume the input 'tools' already conforms to this structure.
            request_payload["tools"] = tools
            if tool_choice:
                request_payload["tool_choice"] = tool_choice
            else:
                request_payload["tool_choice"] = "auto" # Default for OpenAI if tools are provided

        try:
            response = await self.client.chat.completions.create(**request_payload)

            first_choice = response.choices[0]
            message = first_choice.message

            tool_calls = None
            if message.tool_calls:
                # Convert Pydantic ToolCall objects to dictionaries
                tool_calls = [tc.model_dump() for tc in message.tool_calls]

            return {
                "content": message.content,
                "tool_calls": tool_calls,
                "finish_reason": first_choice.finish_reason,
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None,
                "id": response.id,
                "raw_response": response.model_dump_json() # Store the full response as JSON string
            }
        except AuthenticationError as e:
            raise ValueError(f"OpenAI Authentication Error: {e.body.get('message', 'Check API key.')}") from e
        except RateLimitError as e:
            raise RuntimeError(f"OpenAI Rate Limit Exceeded: {e.body.get('message', 'Slow down requests or increase quota.')}") from e
        except BadRequestError as e:
            raise ValueError(f"OpenAI Bad Request Error: {e.body.get('message', 'Invalid request parameters.')}") from e
        except APIError as e:
            raise RuntimeError(f"OpenAI API Error: {e.body.get('message', 'An unexpected API error occurred.')}") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during OpenAI chat completion: {e}") from e

# --- Concrete Anthropic Client ---
try:
    import anthropic
    from anthropic import Anthropic
    from anthropic import APIError as AnthropicAPIError, RateLimitError as AnthropicRateLimitError
except ImportError:
    anthropic = None
    Anthropic = None
    AnthropicAPIError, AnthropicRateLimitError = None, None
    print("Warning: Anthropic library not found. Anthropic client will be unavailable.")

class AnthropicChatClient(BaseLLMClient):
    """
    LLM client for Anthropic models, handling chat completions and tool calls.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        if anthropic is None:
            raise RuntimeError("Anthropic library not installed. Cannot use AnthropicChatClient.")
        super().__init__(model_name, api_key)
        self.client = Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Sends a chat completion request to Anthropic."""
        if temperature is None:
            temperature = settings.DEFAULT_TEMPERATURE
        if max_tokens is None:
            max_tokens = settings.DEFAULT_MAX_TOKENS
        if timeout is None:
            timeout = settings.DEFAULT_TIMEOUT_SECONDS

        anthropic_messages = []
        system_message_content = None
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                # Anthropic's system message is a separate parameter
                system_message_content = content
                continue
            anthropic_messages.append({"role": role, "content": content})

        request_payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            **kwargs
        }

        if system_message_content:
            request_payload["system"] = system_message_content

        if tools:
            # Anthropic expects 'tools' directly, similar to OpenAI's functions
            request_payload["tools"] = tools
            if tool_choice:
                # Map OpenAI-like tool_choice to Anthropic's format
                if isinstance(tool_choice, str):
                    if tool_choice == "auto":
                        request_payload["tool_choice"] = {"type": "auto"}
                    elif tool_choice == "none":
                        # Anthropic default is auto when tools are present, so to disable, don't pass tool_choice
                        pass
                    else: # Fallback for other strings
                        request_payload["tool_choice"] = {"type": "tool", "name": tool_choice}
                elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                    request_payload["tool_choice"] = {"type": "tool", "name": tool_choice["function"]["name"]}
                else:
                    # Pass as is if it's already in Anthropic format or unknown
                    request_payload["tool_choice"] = tool_choice
            else:
                # Default to auto if tools are present and no tool_choice is specified
                request_payload["tool_choice"] = {"type": "auto"}

        try:
            response = await self.client.messages.create(**request_payload)

            tool_calls = []
            content_parts = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function", # Normalize to 'function' type for consistency
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input) # Anthropic returns dict, convert to JSON string
                        }
                    })
                elif block.type == "text":
                    content_parts.append(block.text)

            content = "".join(content_parts) if content_parts else None

            # Anthropic's 'stop_reason' maps to OpenAI's 'finish_reason'
            finish_reason = response.stop_reason if response.stop_reason else "stop"

            return {
                "content": content,
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": finish_reason,
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None,
                "id": response.id,
                "raw_response": response.model_dump_json()
            }
        except AnthropicAPIError as e:
            raise ValueError(f"Anthropic API Error: {e.response.text if e.response else str(e)}") from e
        except AnthropicRateLimitError as e:
            raise RuntimeError(f"Anthropic Rate Limit Exceeded: {e.response.text if e.response else str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during Anthropic chat completion: {e}") from e

# --- Concrete Local LLM Client (Mock/Generic OpenAI-compatible) ---
class LocalChatClient(BaseLLMClient):
    """
    LLM client for local models, typically accessed via an OpenAI-compatible API
    (e.g., Ollama, llama.cpp server, vLLM).
    This implementation uses the OpenAI client for compatibility, assuming the local
    server mimics the OpenAI API.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = "sk-no-key-required", api_base: Optional[str] = None):
        if openai is None:
            raise RuntimeError("OpenAI library not installed. Required for LocalChatClient (via OpenAI-compatible API).")
        super().__init__(model_name, api_key)
        self.api_base = api_base or settings.LOCAL_MODEL_API_BASE
        # Local servers often don't require a real API key, but the client expects one.
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        print(f"Initialized LocalChatClient for model: {self.model_name} at {self.api_base}")

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request to a local OpenAI-compatible LLM server.
        Leverages the OpenAI client library for interaction.
        """
        if temperature is None:
            temperature = settings.DEFAULT_TEMPERATURE
        if max_tokens is None:
            max_tokens = settings.DEFAULT_MAX_TOKENS
        if timeout is None:
            timeout = settings.DEFAULT_TIMEOUT_SECONDS

        request_payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            **kwargs
        }

        if tools:
            request_payload["tools"] = tools
            if tool_choice:
                request_payload["tool_choice"] = tool_choice
            else:
                request_payload["tool_choice"] = "auto"

        try:
            response = await self.client.chat.completions.create(**request_payload)

            first_choice = response.choices[0]
            message = first_choice.message

            tool_calls = None
            if message.tool_calls:
                tool_calls = [tc.model_dump() for tc in message.tool_calls]

            return {
                "content": message.content,
                "tool_calls": tool_calls,
                "finish_reason": first_choice.finish_reason,
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None,
                "id": response.id,
                "raw_response": response.model_dump_json()
            }
        except AuthenticationError as e:
            # Local models might not use an API key, but a missing base_url or malformed request can surface as this.
            raise ValueError(f"Local LLM API (via OpenAI client) Authentication Error: {e.body.get('message', 'Check API base URL or server status.')}") from e
        except RateLimitError as e:
            # Unlikely for local, but could happen if proxying
            raise RuntimeError(f"Local LLM API (via OpenAI client) Rate Limit Exceeded: {e.body.get('message', 'Server busy.')}") from e
        except BadRequestError as e:
            raise ValueError(f"Local LLM API (via OpenAI client) Bad Request Error: {e.body.get('message', 'Invalid request parameters or model not found.')}") from e
        except APIError as e:
            raise RuntimeError(f"Local LLM API (via OpenAI client) Error: {e.body.get('message', 'An unexpected API error occurred with the local server.')}") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during local LLM chat completion: {e}") from e

# --- LLM Client Factory ---
def get_llm_client(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """
    Factory function to get an LLM client instance based on the specified provider.
    If provider, model_name, or api_key are not provided, it falls back to the
    default settings from `config/settings.py`.

    Args:
        provider: The name of the LLM provider (e.g., "openai", "anthropic", "local").
        model_name: The specific model name to use (e.g., "gpt-4o", "claude-3-opus-20240229").
        api_key: The API key for the chosen provider.
        **kwargs: Additional provider-specific arguments (e.g., `api_base` for local models).

    Returns:
        An instance of a class inheriting from `BaseLLMClient`.

    Raises:
        ValueError: If the specified LLM provider is unsupported or required credentials are missing.
        RuntimeError: If necessary libraries for the chosen client are not installed.
    """
    provider = (provider or settings.LLM_PROVIDER).lower()

    if provider == "openai":
        model = model_name or settings.OPENAI_MODEL
        key = api_key or settings.OPENAI_API_KEY
        if not key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass it explicitly to get_llm_client."
            )
        return OpenAIChatClient(model_name=model, api_key=key)
    elif provider == "anthropic":
        model = model_name or settings.ANTHROPIC_MODEL
        key = api_key or settings.ANTHROPIC_API_KEY
        if not key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass it explicitly to get_llm_client."
            )
        return AnthropicChatClient(model_name=model, api_key=key)
    elif provider == "local":
        model = model_name or settings.LOCAL_MODEL_NAME
        api_base = kwargs.get("api_base") or settings.LOCAL_MODEL_API_BASE
        # API key is often not strictly required for local, but passing default for OpenAI client compatibility
        return LocalChatClient(model_name=model, api_key=api_key or "sk-no-key-required", api_base=api_base)
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported providers are 'openai', 'anthropic', 'local'.")

# --- Convenience function for default client ---
async def get_default_llm_client() -> BaseLLMClient:
    """
    Returns the default LLM client instance configured in `config/settings.py`.
    """
    return get_llm_client(
        provider=settings.LLM_PROVIDER,
        model_name=None, # Will use default from settings
        api_key=None,    # Will use default from settings
    )
```