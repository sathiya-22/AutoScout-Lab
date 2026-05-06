import abc
import os
import logging
from typing import List, Dict, Any, Optional

# Set up logging for the module
logger = logging.getLogger(__name__)

class BaseLLMClient(abc.ABC):
    """
    Abstract base class for LLM clients.
    Defines the interface for interacting with various Large Language Models,
    ensuring consistency and enabling easy switching between providers.
    """

    def __init__(self, model_name: str, api_key: str, temperature: float = 0.7, max_tokens: int = 1024):
        """
        Initializes the base LLM client with common configuration parameters.

        Args:
            model_name: The specific model identifier (e.g., "gpt-4", "claude-3-opus-20240229").
            api_key: The API key for authentication with the LLM provider.
            temperature: Controls the randomness of the output. Higher values mean more random.
            max_tokens: The maximum number of tokens to generate in the response.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.debug(f"Initialized BaseLLMClient for model: {self.model_name}")

    @abc.abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Generates a response from the LLM based on a list of messages.
        This method must be implemented by concrete LLM client classes.

        Args:
            messages: A list of message dictionaries, where each dictionary
                      has "role" (e.g., "system", "user", "assistant") and "content" keys.
                      Example: [{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": "What is the capital of France?"}]
            kwargs: Additional parameters specific to the LLM provider's API.

        Returns:
            The generated string response content, or None if an error occurred during generation.
        """
        pass

    @abc.abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Calculates the approximate token count for a given text.
        This is crucial for managing context window limits and cost estimation.
        This method must be implemented by concrete LLM client classes.

        Args:
            text: The input text string to count tokens for.

        Returns:
            The approximate number of tokens in the text.
        """
        pass


class OpenAIChatClient(BaseLLMClient):
    """
    Concrete implementation of an LLM client for OpenAI's Chat Completion models.
    Supports models like GPT-3.5, GPT-4, etc.
    """

    def __init__(self, model_name: str, api_key: str, temperature: float = 0.7, max_tokens: int = 1024):
        """
        Initializes the OpenAI Chat Client.

        Args:
            model_name: The OpenAI model identifier (e.g., "gpt-4o", "gpt-3.5-turbo").
            api_key: Your OpenAI API key.
            temperature: Controls the randomness of the output.
            max_tokens: The maximum number of tokens to generate.
        """
        super().__init__(model_name, api_key, temperature, max_tokens)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAIChatClient initialized for model: {self.model_name}")
        except ImportError:
            logger.error("The 'openai' library is not installed. Please install it with 'pip install openai'.")
            raise
        except Exception as e:
            logger.critical(f"Failed to initialize OpenAI client: {e}")
            raise

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Generates a response using OpenAI's chat completion API.

        Args:
            messages: A list of message dictionaries in OpenAI format.
            kwargs: Additional parameters for the `chat.completions.create` method.

        Returns:
            The generated string response, or None if an API error occurred.
        """
        if not messages:
            logger.warning("Attempted to generate response with empty messages list.")
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            # Check if response or choices are empty/malformed
            if not response.choices or not response.choices[0].message:
                logger.warning(f"OpenAI API returned an empty or malformed response for model {self.model_name}.")
                return None
            return response.choices[0].message.content
        except self.client.APIError as e:
            logger.error(f"OpenAI API Error ({e.status_code}): {e.message}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI API call: {e}")
            return None

    def get_token_count(self, text: str) -> int:
        """
        Calculates the approximate token count for a given text using OpenAI's tiktoken library.

        Args:
            text: The input text string.

        Returns:
            The approximate number of tokens.
        """
        if not text:
            return 0
        try:
            import tiktoken
            # Attempt to get encoding for the specific model, fall back to a common one if not found
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                logger.warning(f"No specific tiktoken encoding for model '{self.model_name}'. "
                               "Falling back to 'cl100k_base' encoding.")
                encoding = tiktoken.get_encoding("cl100k_base") # Common encoding for GPT-4, GPT-3.5-turbo

            return len(encoding.encode(text))
        except ImportError:
            logger.warning("The 'tiktoken' library is not installed. Falling back to a character-based "
                           "approximation (not accurate). Please install it with 'pip install tiktoken'.")
            # Fallback for systems without tiktoken (very rough approximation)
            return len(text) // 4
        except Exception as e:
            logger.warning(f"Could not get token count using tiktoken for model {self.model_name}: {e}. "
                           "Falling back to character-based approximation.")
            return len(text) // 4

# Example of how to add another client (e.g., Anthropic)
# from anthropic import Anthropic, APIStatusError
# class AnthropicChatClient(BaseLLMClient):
#     def __init__(self, model_name: str, api_key: str, temperature: float = 0.7, max_tokens: int = 1024):
#         super().__init__(model_name, api_key, temperature, max_tokens)
#         try:
#             self.client = Anthropic(api_key=self.api_key)
#             logger.info(f"AnthropicChatClient initialized for model: {self.model_name}")
#         except ImportError:
#             logger.error("The 'anthropic' library is not installed. Please install it with 'pip install anthropic'.")
#             raise
#         except Exception as e:
#             logger.critical(f"Failed to initialize Anthropic client: {e}")
#             raise

#     def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
#         if not messages:
#             logger.warning("Attempted to generate response with empty messages list.")
#             return None
#         try:
#             # Anthropic messages format differs slightly, might need reformatting
#             # For simplicity, assuming a conversion utility or specific prompt template handles this.
#             # Or, adapt the messages list here if needed:
#             # System message is separate in Anthropic API
#             system_message = ""
#             api_messages = []
#             for msg in messages:
#                 if msg["role"] == "system":
#                     system_message = msg["content"]
#                 else:
#                     api_messages.append({"role": msg["role"], "content": msg["content"]})
            
#             response = self.client.messages.create(
#                 model=self.model_name,
#                 max_tokens=self.max_tokens,
#                 temperature=self.temperature,
#                 messages=api_messages,
#                 system=system_message if system_message else None,
#                 **kwargs
#             )
#             if not response.content:
#                 logger.warning(f"Anthropic API returned an empty response for model {self.model_name}.")
#                 return None
#             return response.content[0].text
#         except APIStatusError as e:
#             logger.error(f"Anthropic API Error ({e.status_code}): {e.response}")
#             return None
#         except Exception as e:
#             logger.error(f"An unexpected error occurred during Anthropic API call: {e}")
#             return None

#     def get_token_count(self, text: str) -> int:
#         if not text:
#             return 0
#         try:
#             return self.client.count_tokens(text)
#         except Exception as e:
#             logger.warning(f"Could not get token count using Anthropic client for model {self.model_name}: {e}. "
#                            "Falling back to character-based approximation.")
#             return len(text) // 4