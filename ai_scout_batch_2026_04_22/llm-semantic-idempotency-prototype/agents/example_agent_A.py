```python
import logging
import json
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field, ValidationError

# Assuming BaseAgent exists in the same directory for relative import
from .base_agent import BaseAgent

# Assuming LLM proxy interfaces and concrete implementations exist in llm_proxies
from llm_proxies.base_proxy import BaseLLMProxy
from llm_proxies.structured_output_proxy import StructuredOutputProxy

# It's assumed that utils.logging_config.setup_logging() is called at application startup
# to configure the logging system. Modules then just retrieve their logger.
logger = logging.getLogger(__name__)

# Define the expected structured output format for this agent's task using Pydantic
class SummaryAndEntitiesOutput(BaseModel):
    """
    Defines the structured output format for the summarization and entity extraction task.
    """
    summary: str = Field(description="A concise summary of the provided text.")
    entities: List[str] = Field(description="A list of key entities (people, organizations, locations) extracted from the text.")

class ExampleAgentA(BaseAgent):
    """
    ExampleAgentA is a concrete agent implementation responsible for a specific text processing task:
    summarizing input text and extracting key entities.

    It demonstrates how agents interact with LLMs via various LLM proxy types,
    handling both generic string responses (which require manual parsing/validation)
    and directly structured outputs from specialized proxies. This contributes
    to achieving semantic idempotency by enforcing output structure.
    """

    def __init__(self, agent_id: str, llm_proxy: BaseLLMProxy):
        """
        Initializes ExampleAgentA.

        Args:
            agent_id (str): A unique identifier for the agent.
            llm_proxy (BaseLLMProxy): An instance of an LLM proxy to use for LLM interactions.
                                      This can be any concrete implementation of BaseLLMProxy
                                      (e.g., CachingProxy, SeededProxy, StructuredOutputProxy).
                                      The agent's logic adapts based on the proxy's capabilities.
        """
        super().__init__(agent_id, llm_proxy)
        logger.info(f"ExampleAgentA '{self.agent_id}' initialized with LLM Proxy: {type(llm_proxy).__name__}")
        self.task_description = "Summarize the given text into a concise paragraph, extracting key entities."

    def execute(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the agent's primary task: summarizing text and extracting key entities.

        This method encapsulates the agent's core logic, including prompt formulation,
        LLM invocation via the configured proxy, and parsing/validation of the output.

        Args:
            input_data (Dict[str, Any]): The input data for the agent's task.
                                         Expected to contain a 'text' key with the content to process.
            context (Optional[Dict[str, Any]]): Additional context for the execution,
                                                e.g., specific instructions, LLM parameters overrides.

        Returns:
            Dict[str, Any]: A dictionary conforming to the SummaryAndEntitiesOutput schema if successful,
                            or an error dictionary if the task fails at any stage.
        """
        # Validate input data structure
        if not isinstance(input_data, dict) or 'text' not in input_data or not isinstance(input_data['text'], str):
            logger.error(f"Agent '{self.agent_id}' received invalid input_data: {input_data}. "
                         "Expected a dictionary with a 'text' key containing a string.")
            return {"error": "Invalid input format. Expected a dictionary with a 'text' key of type string."}

        text_to_summarize = input_data['text']
        current_context = context if context is not None else {}

        logger.info(f"Agent '{self.agent_id}' starting task: '{self.task_description}' "
                    f"on input text (first 100 chars): '{text_to_summarize[:100]}...'")

        # Construct the base prompt for the LLM
        base_prompt = (
            f"Please summarize the following text into a concise paragraph and "
            f"list the most important entities (people, organizations, locations) found in it.\n\n"
            f"Text:\n\"\"\"\n{text_to_summarize}\n\"\"\"\n\n"
        )

        raw_llm_output: Optional[str] = None # To store raw output for error reporting

        try:
            # Check if the configured proxy supports structured output directly using Pydantic models
            if isinstance(self.llm_proxy, StructuredOutputProxy):
                logger.debug(f"Agent '{self.agent_id}' utilizing StructuredOutputProxy for direct structured output.")
                # Pass the Pydantic model directly to the structured proxy, which will handle
                # prompt injection for schema, LLM call, and Pydantic validation.
                llm_output_obj: SummaryAndEntitiesOutput = self.llm_proxy.invoke(
                    prompt=base_prompt,
                    output_model=SummaryAndEntitiesOutput, # Direct model reference
                    temperature=current_context.get('temperature', 0.0), # Allow context to override
                    max_tokens=current_context.get('max_tokens', 500), # Allow context to override
                    **current_context # Pass any other context parameters
                )
                # StructuredOutputProxy returns a validated Pydantic object, convert to dict
                return llm_output_obj.model_dump()
            else:
                logger.debug(f"Agent '{self.agent_id}' using generic BaseLLMProxy, expecting JSON string output.")
                # For generic proxies, explicitly instruct the LLM to output JSON in the prompt
                json_prompt = base_prompt + (
                    f"Output should be strictly in JSON format with two keys: 'summary' (string) "
                    f"and 'entities' (list of strings).\n"
                    f"Example: {{ \"summary\": \"The text describes...\", \"entities\": [\"Entity1\", \"Entity2\"] }}"
                )
                raw_llm_output = self.llm_proxy.invoke(
                    prompt=json_prompt,
                    temperature=current_context.get('temperature', 0.0),
                    max_tokens=current_context.get('max_tokens', 500),
                    **current_context
                )
                logger.debug(f"Agent '{self.agent_id}' received raw LLM output: {raw_llm_output}")

                # Attempt to parse the raw string output as JSON
                parsed_dict = json.loads(raw_llm_output)
                
                # Validate the parsed dictionary against the Pydantic model
                llm_output_obj = SummaryAndEntitiesOutput.model_validate(parsed_dict)
                
                logger.info(f"Agent '{self.agent_id}' successfully processed and validated LLM output.")
                return llm_output_obj.model_dump()

        except json.JSONDecodeError as jde:
            # Handles cases where the LLM response is not valid JSON
            logger.error(f"Agent '{self.agent_id}' failed to parse LLM output as JSON: {jde}. "
                         f"Raw output: '{raw_llm_output}'")
            return {"error": "Failed to parse LLM response as JSON.", "raw_output": raw_llm_output}
        except ValidationError as ve:
            # Handles cases where JSON is valid but does not match the Pydantic schema
            logger.error(f"Agent '{self.agent_id}' received malformed LLM output (Pydantic validation error): {ve}. "
                         f"Raw output: '{raw_llm_output}'")
            return {"error": f"LLM response did not conform to the expected schema: {ve}", "raw_output": raw_llm_output}
        except Exception as e:
            # Catches any other unexpected errors during LLM invocation or processing
            logger.exception(f"Agent '{self.agent_id}' encountered an unexpected error during LLM invocation or processing.")
            return {"error": f"An unexpected error occurred during agent execution: {str(e)}", "raw_output": raw_llm_output}
```