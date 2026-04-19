```python
import json
from typing import Dict, Any

class ContextInjector:
    """
    Injects structured contextual information into LLM prompts to reduce variability
    and anchor decisions to specific data points or internal states.

    This class supports various injection methods and formatting options to
    ensure the context is clearly presented to the LLM.
    """

    def _format_context_plain(self, context: Dict[str, Any]) -> str:
        """
        Formats a dictionary of context into a plain, human/LLM-readable string.
        Each key-value pair is presented on its own line, enclosed within
        distinct markers.
        """
        if not context:
            return ""

        formatted_lines = ["[CONTEXTUAL_ANCHORS_START]"]
        for key, value in context.items():
            # Convert non-string values to string for plain representation
            formatted_lines.append(f"{key}: {str(value)}")
        formatted_lines.append("[CONTEXTUAL_ANCHORS_END]")
        # Ensure consistent separation from the main prompt content
        return "\n".join(formatted_lines) + "\n\n"

    def _format_context_json(self, context: Dict[str, Any]) -> str:
        """
        Formats a dictionary of context into a JSON string, enclosed within
        markdown code block syntax (```json). This helps LLMs parse the
        structured data more reliably.
        """
        if not context:
            return ""
        try:
            json_str = json.dumps(context, indent=2)
            # Use markdown code block for better LLM parsing of JSON
            return f"```json\n{json_str}\n```\n\n"
        except TypeError as e:
            # If context contains non-serializable types, fall back to plain format.
            # In a production environment, this might also trigger a log warning.
            print(f"Warning: Could not serialize context to JSON: {e}. Falling back to plain format.")
            return self._format_context_plain(context)

    def inject_context(self,
                       prompt: str,
                       context: Dict[str, Any],
                       method: str = "prepend",
                       format_type: str = "plain") -> str:
        """
        Injects structured contextual information into an LLM prompt.

        Args:
            prompt (str): The original LLM prompt string.
            context (Dict[str, Any]): A dictionary of contextual information
                                       (e.g., unique IDs, hashes, execution parameters,
                                       previous states). This context will be serialized
                                       and inserted into the prompt.
            method (str): The injection method to use:
                          - "prepend": Inserts the formatted context at the beginning of the prompt.
                          - "append": Inserts the formatted context at the end of the prompt.
                          - "replace_placeholder": Replaces a specific placeholder
                                                   (e.g., "{context}") within the prompt with
                                                   the formatted context.
            format_type (str): The format for the injected context:
                               - "plain": Simple key-value pairs with delimiters (human/LLM readable).
                               - "json": JSON formatted string within a markdown code block.

        Returns:
            str: The augmented prompt with the injected contextual information.

        Raises:
            TypeError: If `prompt` is not a string or `context` is not a dictionary.
            ValueError: If an unsupported injection method or format type is provided,
                        or if the placeholder is not found for 'replace_placeholder' method.
        """
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")
        if not isinstance(context, dict):
            raise TypeError("Context must be a dictionary.")

        if not context:
            return prompt # No context to inject, return original prompt

        formatted_context_str: str

        if format_type == "plain":
            formatted_context_str = self._format_context_plain(context)
        elif format_type == "json":
            formatted_context_str = self._format_context_json(context)
        else:
            raise ValueError(f"Unsupported context format type: '{format_type}'. "
                             "Supported types are 'plain', 'json'.")

        if method == "prepend":
            return formatted_context_str + prompt
        elif method == "append":
            # Add extra newlines for clear separation before appending
            return prompt + "\n\n" + formatted_context_str.rstrip('\n')
        elif method == "replace_placeholder":
            placeholder = "{context}"
            if placeholder not in prompt:
                raise ValueError(
                    f"Placeholder '{placeholder}' not found in prompt for 'replace_placeholder' method. "
                    "Ensure your prompt includes '{context}' where you want the context injected."
                )
            # Remove trailing newlines from the formatted string if replacing,
            # to allow the prompt template to manage spacing.
            return prompt.replace(placeholder, formatted_context_str.rstrip('\n'))
        else:
            raise ValueError(f"Unsupported injection method: '{method}'. "
                             "Supported methods are 'prepend', 'append', 'replace_placeholder'.")

```