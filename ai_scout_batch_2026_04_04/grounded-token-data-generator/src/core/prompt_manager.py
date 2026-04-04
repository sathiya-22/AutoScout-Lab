```python
import logging
from typing import List, Union, Dict, Any

# Configure logging for the module
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Add a NullHandler to prevent "No handlers could be found for logger" messages
    # when the root logger is not configured.
    # In a real application, a proper logging configuration would be set up
    # at the application entry point.
    logger.addHandler(logging.NullHandler())


class PromptManager:
    """
    Manages and centralizes prompt templates for generating and refining
    linguistic supervision for novel tokens.

    This class provides a structured way to store, retrieve, and format
    various prompt templates, enabling easy experimentation and optimization
    of prompting strategies for different token types or domains.
    """

    _templates: Dict[str, str] = {
        "initial_description_generation": """
        You are an expert AI assistant specializing in natural language understanding and vocabulary expansion.
        Your primary task is to generate a concise, informative, and semantically grounded natural language description (1-3 sentences) for a novel vocabulary token.
        This description will be used to help a large language model understand the token's meaning, typical usage, and semantic relations within its domain.

        ---
        Novel Token: "{token}"
        Context/Domain: "{context}"
        Relevant Knowledge:
        {knowledge_snippets}
        ---

        Considering the "Novel Token", its "Context/Domain", and the "Relevant Knowledge", craft a description that clearly defines what the token represents, its key attributes, or its primary function.
        Ensure the description is unambiguous, avoids jargon where possible, and is suitable for an LLM to learn from.
        Focus on providing a direct and helpful definition.
        """,

        "description_refinement": """
        You are an expert AI assistant tasked with refining an existing natural language description for a novel vocabulary token.
        The goal is to enhance its clarity, completeness, semantic grounding, and overall quality for LLM training.

        ---
        Novel Token: "{token}"
        Existing Description: "{existing_description}"
        Context/Domain: "{context}"
        Additional Knowledge/Context:
        {knowledge_snippets}
        Specific Human Feedback (if available): "{human_feedback}"
        ---

        Please critically review the "Existing Description" and refine it based on the "Additional Knowledge/Context" and any "Specific Human Feedback".
        The refined description should be 1-3 sentences, more accurate, comprehensive, and better grounded in the provided information.
        If no specific feedback or additional knowledge is provided, focus on improving the fluency, conciseness, or overall explanatory power of the existing description.
        """,

        "semantic_grounding_assessment": """
        You are an AI assistant specialized in evaluating linguistic descriptions against factual knowledge.
        Your task is to assess how well the given description for a novel token aligns with the provided key knowledge.

        ---
        Novel Token: "{token}"
        Description to Assess: "{description}"
        Key Knowledge for Grounding:
        {knowledge_snippets}
        ---

        Evaluate the "Description to Assess" against the "Key Knowledge for Grounding".
        Provide a concise judgment: "ACCURATE", "PARTIALLY_ACCURATE", or "INACCURATE".
        If the judgment is "PARTIALLY_ACCURATE" or "INACCURATE", explain any significant discrepancies, omissions, or misinterpretations.
        Also, suggest specific, actionable improvements to make the description fully aligned and comprehensive based on the provided knowledge.
        """,
        # Additional prompt templates can be added here for other tasks,
        # such as generating examples of usage, identifying antonyms/synonyms,
        # or domain-specific description generation.
    }

    def get_template(self, template_name: str) -> str:
        """
        Retrieves a prompt template by its name.

        Args:
            template_name: The name of the prompt template to retrieve
                           (e.g., 'initial_description_generation').

        Returns:
            The string content of the prompt template.

        Raises:
            ValueError: If the `template_name` is not found in the defined templates.
        """
        if template_name not in self._templates:
            logger.error(f"Attempted to retrieve unknown prompt template: '{template_name}'")
            raise ValueError(f"Prompt template '{template_name}' not found.")
        return self._templates[template_name]

    def _format_knowledge_snippets(self, snippets: Union[List[str], str, None]) -> str:
        """
        Helper method to format a list of knowledge snippets (or a single string)
        into a readable, bullet-point string for inclusion in the prompt.
        If no snippets are provided, a default message is returned.

        Args:
            snippets: A list of strings, a single string, or None.

        Returns:
            A formatted string suitable for embedding in a prompt.
        """
        if isinstance(snippets, list):
            if not snippets:
                return "No specific knowledge provided."
            # Ensure each snippet is converted to string and stripped before joining
            return "\n".join([f"- {str(s).strip()}" for s in snippets if str(s).strip()])
        elif isinstance(snippets, str) and snippets.strip():
            # If it's already a non-empty string, use it directly (assume it's pre-formatted)
            return snippets.strip()
        else:
            return "No specific knowledge provided."

    def format_prompt(self, template_name: str, **kwargs: Any) -> str:
        """
        Formats a prompt template with provided keyword arguments.

        This method retrieves a template by `template_name` and populates its
        placeholders using the values from `kwargs`. Sensible default values
        are provided for commonly optional parameters to ensure robust formatting.

        Args:
            template_name: The name of the prompt template to use.
            **kwargs: Arbitrary keyword arguments to fill in the template placeholders.
                      Common arguments and their expected types include:
                      - `token` (str, required by most templates): The novel vocabulary token.
                      - `context` (str): The domain or surrounding context for the token.
                      - `knowledge_snippets` (list[str] or str): Relevant factual snippets or information.
                      - `existing_description` (str): An existing description to be refined.
                      - `human_feedback` (str): Specific feedback from a human annotator.
                      - `description` (str): A description to be assessed (for grounding checks).

        Returns:
            The formatted prompt string ready for an LLM.

        Raises:
            ValueError: If the `template_name` is not found, or if formatting fails
                        due to missing mandatory keys that do not have a default.
        """
        template = self.get_template(template_name)

        # Prepare kwargs with sensible defaults for common optional fields.
        # This prevents KeyError during .format() for fields that might not always be present
        # but are optionally included in some templates.
        processed_kwargs = {
            "token": kwargs.get("token", "[MISSING_TOKEN]"), # Token is usually mandatory, but provide default for safety
            "context": kwargs.get("context", "general domain or N/A"),
            "knowledge_snippets": self._format_knowledge_snippets(kwargs.get("knowledge_snippets")),
            "existing_description": kwargs.get("existing_description", "N/A"),
            "human_feedback": kwargs.get("human_feedback", "N/A"),
            "description": kwargs.get("description", "N/A"), # Used for semantic_grounding_assessment
            # Add other common optional fields with their defaults here
        }

        # Override defaults with any values explicitly provided in kwargs.
        # This allows users to provide empty strings explicitly if that's desired.
        final_kwargs = {**processed_kwargs, **kwargs}

        try:
            formatted_prompt = template.format(**final_kwargs)
            logger.debug(f"Successfully formatted prompt '{template_name}' for token '{final_kwargs.get('token')}'")
            return formatted_prompt
        except KeyError as e:
            logger.error(f"Failed to format prompt '{template_name}'. Missing mandatory placeholder: '{e}'. "
                         f"Available keys for formatting: {list(final_kwargs.keys())}")
            raise ValueError(f"Failed to format prompt '{template_name}'. A required parameter '{e}' was not provided or had no default.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while formatting prompt '{template_name}'.") # Use exception for full traceback
            raise ValueError(f"An unexpected error occurred during prompt formatting: {e}")

```