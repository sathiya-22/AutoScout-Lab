```python
import pydantic
from typing import Dict, Any, Tuple

class SchemaValidationError(Exception):
    """
    Custom exception raised for errors during tool call schema validation.
    Provides a clear message and optional details about the validation failure.
    """
    def __init__(self, message: str, details: Any = None):
        super().__init__(message)
        self.details = details

class SchemaValidator:
    """
    The SchemaValidator performs deterministic checks against a tool's defined schema.
    It ensures correct function names, argument types, required parameters, and
    adherence to specified value constraints (e.g., ranges, regex patterns)
    by leveraging Pydantic models. This acts as a formal verification component
    ensuring syntactic and basic semantic correctness of proposed tool calls.
    """

    def validate_tool_call(self, proposed_tool_call: Dict[str, Any], tool_schemas: Dict[str, pydantic.BaseModel]) -> Tuple[str, pydantic.BaseModel]:
        """
        Validates a proposed tool call against a dictionary of Pydantic tool schemas.

        This method is the first line of defense for validating LLM-generated tool calls.
        It strictly adheres to the defined Pydantic schemas, ensuring that:
        1. The tool name exists among the registered tools.
        2. All required arguments for the specified tool are present.
        3. All provided arguments have the correct data types.
        4. Any Pydantic-defined value constraints (e.g., Field(gt=0, regex=...)) are met.

        Args:
            proposed_tool_call: A dictionary representing the tool call proposed by the
                                primary agent, expected to have keys "tool_name" (str)
                                and "args" (dict).
                                Example: {"tool_name": "get_weather", "args": {"location": "London"}}
            tool_schemas: A dictionary where keys are tool names (str) and values are
                          Pydantic BaseModel classes that define the expected arguments
                          schema for each respective tool.

        Returns:
            A tuple containing:
            - The name of the validated tool (str).
            - An instance of the corresponding Pydantic BaseModel for the tool's
              arguments, with all values type-coerced and validated.

        Raises:
            SchemaValidationError: If the proposed tool call does not conform to any
                                   aspect of the defined schema, including:
                                   - Missing 'tool_name' or 'args' keys.
                                   - Incorrect types for 'tool_name' or 'args'.
                                   - An unknown 'tool_name'.
                                   - Mismatch between 'args' and the tool's Pydantic schema
                                     (e.g., missing required fields, incorrect argument types,
                                     or invalid argument values).
        """
        if not isinstance(proposed_tool_call, dict):
            raise SchemaValidationError(
                f"Proposed tool call must be a dictionary, but got {type(proposed_tool_call).__name__}."
            )

        tool_name = proposed_tool_call.get("tool_name")
        args = proposed_tool_call.get("args")

        # Basic structural validation
        if not tool_name:
            raise SchemaValidationError("Proposed tool call is missing the 'tool_name' key.")
        if not isinstance(tool_name, str):
            raise SchemaValidationError(
                f"Tool name must be a string, but got {type(tool_name).__name__} for '{tool_name}'."
            )
        if args is None:  # `args` can be an empty dict, but not None
            raise SchemaValidationError(
                f"Proposed tool call is missing the 'args' key or 'args' is None for tool '{tool_name}'."
            )
        if not isinstance(args, dict):
            raise SchemaValidationError(
                f"Tool arguments must be a dictionary, but got {type(args).__name__} for tool '{tool_name}'."
            )

        # Check if the tool name exists in the provided schemas
        if tool_name not in tool_schemas:
            available_tools = list(tool_schemas.keys())
            raise SchemaValidationError(
                f"Unknown tool name '{tool_name}'. Available tools are: {', '.join(available_tools) or 'None'}."
            )

        tool_args_schema = tool_schemas[tool_name]

        try:
            # Pydantic's parse_obj_as will validate the `args` dictionary against
            # the specified BaseModel, handling type coercion, default values,
            # required fields, and field-level validators.
            validated_args = pydantic.parse_obj_as(tool_args_schema, args)
            return tool_name, validated_args
        except pydantic.ValidationError as e:
            # Catch Pydantic-specific validation errors and re-raise them as
            # our custom SchemaValidationError, including detailed error messages.
            raise SchemaValidationError(
                f"Schema validation failed for tool '{tool_name}' due to argument mismatch.",
                details=e.errors()
            )
        except Exception as e:
            # Catch any other unexpected errors during the validation process
            raise SchemaValidationError(
                f"An unexpected error occurred during schema validation for tool '{tool_name}'.",
                details=str(e)
            )
```