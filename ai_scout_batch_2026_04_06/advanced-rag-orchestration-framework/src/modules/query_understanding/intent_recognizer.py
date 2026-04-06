```python
import logging
import json
from typing import List, Dict, Any, Optional

from ...core.interfaces import QueryProcessor
from ...core.models import Query, Intent, IntentType, IntentParameter
from ...providers.llm_provider import LLMProvider, LLMResponse, GenerationConfig
from ...config import Config

logger = logging.getLogger(__name__)

class IntentRecognizer(QueryProcessor):
    """
    The IntentRecognizer module analyzes a user's query to identify its primary intent
    and extract relevant parameters. It leverages an LLM to perform this classification
    and structuring of information, enhancing the `Query` object with recognized intent.
    """
    def __init__(self, llm_provider: LLMProvider, config: Optional[Config] = None):
        """
        Initializes the IntentRecognizer with an LLM provider and configuration.

        Args:
            llm_provider (LLMProvider): An instance of an LLMProvider to use for intent classification.
            config (Optional[Config]): Configuration object for settings like prompts and model names.
        """
        if not isinstance(llm_provider, LLMProvider):
            raise TypeError("llm_provider must be an instance of LLMProvider.")

        self.llm_provider = llm_provider
        self.config = config or Config() # Use provided config or a default instance

        # Load prompts and LLM settings from config with sensible defaults
        self._default_system_prompt = self.config.get(
            "intent_recognizer_system_prompt", self._get_default_system_prompt()
        )
        self._default_user_prompt_template = self.config.get(
            "intent_recognizer_user_prompt_template", self._get_default_user_prompt_template()
        )
        self._default_model_name = self.config.get(
            "intent_recognizer_llm_model", "gpt-3.5-turbo" # A common default LLM
        )
        self._default_temperature = self.config.get(
            "intent_recognizer_llm_temperature", 0.1 # Low temperature for factual extraction
        )

    def _get_default_system_prompt(self) -> str:
        """
        Defines the default system prompt for the LLM to guide intent recognition.
        This prompt instructs the LLM on expected intent types, parameters, and output format.
        """
        return """You are an advanced intent recognition system. Your task is to analyze user queries, identify the primary intent, and extract relevant parameters.
        Respond ONLY with a JSON object. Do not include any conversational text or explanations.

        Here are the supported intent types and their expected parameters:
        - `SEARCH`: For general information retrieval.
          Parameters: `keywords` (string, required), `filters` (optional, list of key-value objects, e.g., [{"field": "category", "value": "electronics"}]).
        - `SUMMARIZE`: When the user asks to summarize a topic or document.
          Parameters: `topic` (string, required), `length` (optional, string, e.g., "short", "medium", "long").
        - `COMPARE`: When the user asks to compare two or more entities.
          Parameters: `entities` (list of strings, required), `aspects` (optional, list of strings).
        - `DEFINITION`: When the user asks for the definition of a term.
          Parameters: `term` (string, required).
        - `NAVIGATE`: When the user intends to go to a specific section or page.
          Parameters: `destination` (string, required, e.g., "settings page", "checkout").
        - `GENERATE_CODE`: When the user asks to generate code.
          Parameters: `language` (string, required), `task` (string, required), `constraints` (optional, list of strings).
        - `GREETING`: Simple greetings.
          Parameters: None.
        - `UNKNOWN`: If none of the above intents match.
          Parameters: `original_query` (string, required).

        If a parameter is not present or cannot be extracted, omit it from the JSON.
        The JSON output MUST always have two top-level keys: "intent_type" (string) and "parameters" (object).
        Example for 'SEARCH' intent: {"intent_type": "SEARCH", "parameters": {"keywords": "latest AI research", "filters": [{"field": "year", "value": "2023"}]}}
        Example for 'GREETING' intent: {"intent_type": "GREETING", "parameters": {}}
        Example for 'UNKNOWN' intent: {"intent_type": "UNKNOWN", "parameters": {"original_query": "What's the meaning of life?"}}
        """

    def _get_default_user_prompt_template(self) -> str:
        """
        Defines the default template for incorporating the user's query into the LLM prompt.
        """
        return "User Query: \"{query}\""

    async def process(self, query: Query) -> Query:
        """
        Processes a `Query` object to identify its intent and extract parameters using an LLM.
        The recognized intent and parameters are then added to the `query.intent` field.

        Args:
            query (Query): The input Query object containing the user's raw text.

        Returns:
            Query: The updated Query object with the `intent` field populated.
        """
        logger.info(f"Attempting to recognize intent for query ID '{query.id}': '{query.text[:100]}...'")

        # Construct the messages for the LLM call
        messages = [
            {"role": "system", "content": self._default_system_prompt},
            {"role": "user", "content": self._default_user_prompt_template.format(query=query.text)}
        ]

        # Configure LLM generation for structured JSON output
        generation_config = GenerationConfig(
            model=self._default_model_name,
            temperature=self._default_temperature,
            response_format={"type": "json_object"} # Crucial for reliable JSON output
        )

        try:
            llm_response: LLMResponse = await self.llm_provider.generate(
                messages=messages,
                config=generation_config
            )

            if not llm_response.text:
                raise ValueError("LLM returned an empty response for intent recognition.")

            intent_data = self._parse_llm_response(llm_response.text)
            recognized_intent = self._create_intent_object(intent_data, query.text)
            query.intent = recognized_intent
            logger.info(
                f"Intent recognized for query ID '{query.id}': {recognized_intent.type.value} "
                f"with parameters: {[(p.name, p.value) for p in recognized_intent.parameters]}"
            )

        except Exception as e:
            logger.error(
                f"Error recognizing intent for query ID '{query.id}', text '{query.text[:100]}...': {e}",
                exc_info=True
            )
            # Fallback: Assign UNKNOWN intent on any processing error
            query.intent = Intent(
                type=IntentType.UNKNOWN,
                parameters=[IntentParameter(name="original_query", value=query.text)]
            )
            logger.warning(
                f"Fallback to UNKNOWN intent for query ID '{query.id}' due to error: {e.__class__.__name__}."
            )

        return query

    def _parse_llm_response(self, json_string: str) -> Dict[str, Any]:
        """
        Parses the JSON string returned by the LLM.

        Args:
            json_string (str): The raw JSON string from the LLM.

        Returns:
            Dict[str, Any]: The parsed JSON as a dictionary.

        Raises:
            ValueError: If the string is not valid JSON or not a JSON object.
        """
        try:
            data = json.loads(json_string)
            if not isinstance(data, dict):
                raise ValueError("LLM response is not a valid JSON object.")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM response. Response snippet: '{json_string[:200]}...'. Error: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during JSON parsing of LLM response: {e}")
            raise ValueError(f"Error parsing LLM response: {e}") from e

    def _create_intent_object(self, intent_data: Dict[str, Any], original_query_text: str) -> Intent:
        """
        Creates an `Intent` Pydantic object from the parsed LLM response dictionary.

        Args:
            intent_data (Dict[str, Any]): The dictionary parsed from the LLM's JSON response.
            original_query_text (str): The original text of the query, used for fallback `UNKNOWN` intent.

        Returns:
            Intent: A Pydantic `Intent` model.
        """
        intent_type_str = intent_data.get("intent_type", "UNKNOWN").upper()

        # Safely convert string to IntentType enum
        try:
            intent_type = IntentType[intent_type_str]
        except KeyError:
            logger.warning(
                f"LLM returned an unrecognized intent type '{intent_type_str}'. Defaulting to UNKNOWN."
            )
            intent_type = IntentType.UNKNOWN

        params_data = intent_data.get("parameters", {})
        parameters: List[IntentParameter] = []

        if intent_type == IntentType.UNKNOWN:
            # For UNKNOWN intent, always include original query as a parameter
            parameters.append(IntentParameter(name="original_query", value=original_query_text))
        elif isinstance(params_data, dict):
            # Process parameters only if params_data is a dictionary
            for key, value in params_data.items():
                # Basic validation: ensure parameter values are serializable types
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    parameters.append(IntentParameter(name=key, value=value))
                else:
                    logger.warning(
                        f"Unsupported parameter value type for key '{key}': {type(value)}. Skipping this parameter."
                    )
        elif params_data is not None:
            logger.warning(
                f"Expected 'parameters' to be a dictionary or empty, but got type {type(params_data)}. "
                "Skipping parameter parsing for non-dictionary type."
            )

        return Intent(type=intent_type, parameters=parameters)
```