```python
import json
import logging
from typing import List, Optional, Dict, Any

# Assuming these modules and classes exist based on the architecture notes
from src.utils.llm_interface import LLMClient
from src.prompting.prompt_manager import PromptManager
from src.tools.tool_definitions import get_tool_definitions
from src.utils.tool_schemas import ToolCall, ToolDefinition  # ToolDefinition for type hinting

logger = logging.getLogger(__name__)

# Basic logging configuration if running standalone, otherwise main.py will configure it
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PrimaryAgent:
    """
    The PrimaryAgent is the initial LLM agent responsible for interpreting user requests
    and proposing tool calls. It leverages sophisticated prompting techniques to maximize
    its initial accuracy in identifying and structuring tool invocations.

    It interacts with an LLM via LLMClient and uses PromptManager to construct
    well-formatted prompts, including tool definitions and few-shot examples.
    """
    def __init__(self, llm_client: LLMClient, prompt_manager: PromptManager):
        """
        Initializes the PrimaryAgent with an LLM client and a prompt manager.

        Args:
            llm_client: An instance of LLMClient for interacting with the underlying LLM.
            prompt_manager: An instance of PromptManager to generate structured prompts
                            for the LLM.
        """
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        # Load tool definitions once at initialization to avoid repeated I/O
        try:
            self.tool_definitions: List[ToolDefinition] = get_tool_definitions()
            logger.info(f"PrimaryAgent initialized with {len(self.tool_definitions)} tool definitions.")
        except Exception as e:
            logger.error(f"Failed to load tool definitions during PrimaryAgent initialization: {e}")
            self.tool_definitions = [] # Ensure it's always an empty list on failure

    def propose_tool_call(self, user_query: str) -> Optional[ToolCall]:
        """
        Interprets a user query and attempts to propose a tool call based on
        available tools and advanced prompting strategies.

        This method:
        1. Prepares the available tool definitions for the prompt.
        2. Constructs a comprehensive prompt using the PromptManager, including
           system instructions, tool schemas, and potentially few-shot examples.
        3. Calls the underlying LLM with the generated prompt.
        4. Parses and validates the LLM's response, expecting a JSON object
           that conforms to the ToolCall schema.

        Args:
            user_query: The user's natural language request (e.g., "What's the weather in London?").

        Returns:
            An Optional[ToolCall] object.
            Returns a ToolCall instance if the LLM successfully proposes a valid
            tool call that can be parsed.
            Returns None if the LLM's response is empty, unparseable, or does
            not conform to the ToolCall schema.
        """
        if not self.tool_definitions:
            logger.warning("No tool definitions available. PrimaryAgent cannot propose tool calls.")
            return None

        # Convert tool definitions to a JSON-serializable dictionary format.
        # This is suitable for embedding into the prompt text, allowing the LLM
        # to understand the structure and purpose of each tool.
        # Assuming ToolDefinition is a Pydantic model and has .model_dump() method.
        formatted_tool_defs = [tool.model_dump(mode='json') for tool in self.tool_definitions]

        # Generate the full prompt messages using the prompt manager.
        # The prompt manager orchestrates the construction of the prompt,
        # embedding system instructions, few-shot examples (if configured),
        # the available tool schemas, and the current user query.
        full_prompt_messages = []
        try:
            full_prompt_messages = self.prompt_manager.generate_primary_agent_prompt(
                user_query=user_query,
                tool_definitions=formatted_tool_defs
            )
            logger.debug(f"Generated prompt messages for query '{user_query}': {full_prompt_messages}")
        except Exception as e:
            logger.error(f"Error generating prompt messages for query '{user_query}': {e}")
            return None

        # Call the underlying LLM with the constructed prompt.
        llm_response_content: Optional[str] = None
        try:
            llm_response_content = self.llm_client.chat_completion(messages=full_prompt_messages)
            if not llm_response_content:
                logger.warning("LLM returned an empty response for query: '%s'. No tool call proposed.", user_query)
                return None
            logger.debug(f"LLM raw response for query '{user_query}': {llm_response_content}")

        except Exception as e:
            logger.error(f"Error during LLM chat completion for query '{user_query}': {e}")
            return None

        # Parse and validate the LLM's response.
        # The LLM is instructed via the prompt to output a JSON string
        # representing the proposed tool call. This JSON might sometimes be
        # wrapped in markdown code blocks (e.g., ```json...``` or ```...```).
        try:
            json_str = llm_response_content.strip()

            # Attempt to strip markdown code block wrappers
            if json_str.startswith("```json") and json_str.endswith("```"):
                json_str = json_str[len("```json"): -len("```")].strip()
            elif json_str.startswith("```") and json_str.endswith("```"): # Generic code block
                 json_str = json_str[len("```"): -len("```")].strip()

            parsed_response: Dict[str, Any] = json.loads(json_str)

            # Use Pydantic to validate the structure of the parsed JSON
            # against the expected ToolCall schema. This ensures the output
            # is well-formed before further processing.
            tool_call = ToolCall(**parsed_response)
            logger.info(f"Successfully parsed LLM response into ToolCall for query '{user_query}'.")
            logger.debug(f"Proposed ToolCall: {tool_call.model_dump_json()}")
            return tool_call

        except json.JSONDecodeError as e:
            logger.warning(
                "LLM response for query '%s' is not valid JSON. No tool call proposed. Error: %s | Response: '%s'",
                user_query, e, llm_response_content
            )
            return None
        except Exception as e: # Catches Pydantic validation errors or other unexpected issues
            logger.warning(
                "Failed to create ToolCall object from LLM response for query '%s' (schema mismatch or other error). Error: %s | Response: '%s'",
                user_query, e, llm_response_content
            )
            return None

```