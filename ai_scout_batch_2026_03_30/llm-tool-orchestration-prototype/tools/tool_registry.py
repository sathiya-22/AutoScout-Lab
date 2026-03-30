from typing import Dict, Callable, Any, List, Optional, Type
from pydantic import BaseModel, Field

# Placeholder for ToolDefinition. In a full system, this would typically be imported
# from tools.tool_definitions to keep definitions separate.
# However, as per the instruction "Return ONLY the code for this file", it must be
# self-contained within this file.
class ToolDefinition(BaseModel):
    """
    Defines the structure and metadata for an external tool.
    """
    name: str = Field(..., description="Unique name of the tool.")
    description: str = Field(..., description="Description of what the tool does.")
    input_schema: Type[BaseModel] = Field(..., description="Pydantic schema for the tool's input arguments.")
    func: Callable[..., Any] = Field(..., exclude=True, description="The actual Python function to call.")

    class Config:
        arbitrary_types_allowed = True # Allows 'func' to be a callable without Pydantic trying to validate its internal structure aggressively.

class ToolRegistry:
    """
    Manages the registration, retrieval, and metadata of all available tools.
    This class is implemented as a singleton to ensure a single, consistent
    source of truth for available tools across the application.
    """

    _instance: Optional['ToolRegistry'] = None
    _tools: Dict[str, ToolDefinition] = {}

    def __new__(cls):
        """Ensures that only one instance of ToolRegistry is created (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            cls._tools = {}  # Initialize the tools dictionary when the first instance is created
        return cls._instance

    def register_tool(self, tool_definition: ToolDefinition):
        """
        Registers a tool with the registry.

        Args:
            tool_definition: An instance of ToolDefinition containing tool metadata
                             and the callable function.

        Raises:
            TypeError: If the provided tool_definition is not an instance of ToolDefinition,
                       or if input_schema is not a Pydantic BaseModel, or if func is not callable.
            ValueError: If a tool with the same name is already registered, or if the tool name
                        is invalid.
        """
        if not isinstance(tool_definition, ToolDefinition):
            raise TypeError(
                f"Invalid tool_definition type. Expected ToolDefinition, got {type(tool_definition).__name__}."
            )

        if not tool_definition.name or not isinstance(tool_definition.name, str):
            raise ValueError("Tool name must be a non-empty string.")

        if tool_definition.name in self._tools:
            raise ValueError(f"Tool '{tool_definition.name}' is already registered.")

        if not issubclass(tool_definition.input_schema, BaseModel):
             raise TypeError(
                f"Input schema for tool '{tool_definition.name}' must be a Pydantic BaseModel, "
                f"got {tool_definition.input_schema.__name__}."
            )

        if not callable(tool_definition.func):
            raise TypeError(f"The 'func' attribute of tool '{tool_definition.name}' must be a callable.")

        self._tools[tool_definition.name] = tool_definition

    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Retrieves the ToolDefinition for a given tool name.

        Args:
            tool_name: The name of the tool to retrieve.

        Returns:
            The ToolDefinition instance if found, otherwise None.
        """
        return self._tools.get(tool_name)

    def get_callable_tool(self, tool_name: str) -> Optional[Callable[..., Any]]:
        """
        Retrieves the callable function for a given tool name.

        Args:
            tool_name: The name of the tool.

        Returns:
            The callable function if found, otherwise None.
        """
        tool_def = self.get_tool_definition(tool_name)
        return tool_def.func if tool_def else None

    def get_tool_names(self) -> List[str]:
        """
        Returns a list of all registered tool names.
        """
        return list(self._tools.keys())

    def get_all_tool_metadata(self) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries, each representing the metadata
        (name, description, input_schema as JSON schema dictionary) for a registered tool.
        This structured output is ideal for providing context to LLMs, allowing them
        to understand available tools and their required arguments.
        """
        metadata_list = []
        for tool_name, tool_def in self._tools.items():
            try:
                # Use model_json_schema() for Pydantic v2+ to get the JSON schema representation
                if hasattr(tool_def.input_schema, 'model_json_schema'):
                    input_schema_dict = tool_def.input_schema.model_json_schema()
                elif hasattr(tool_def.input_schema, 'schema'): # Fallback for Pydantic v1
                    input_schema_dict = tool_def.input_schema.schema()
                else:
                    input_schema_dict = {"error": "Input schema does not support JSON schema generation."}
            except Exception as e:
                # Catch any other exceptions during schema retrieval
                input_schema_dict = {"error": f"Failed to retrieve schema: {e}"}

            metadata_list.append({
                "name": tool_def.name,
                "description": tool_def.description,
                "input_schema": input_schema_dict
            })
        return metadata_list

    def clear_registry(self):
        """
        Clears all registered tools from the registry.
        This method is primarily useful for testing or resetting the environment.
        """
        self._tools.clear()

# Create a singleton instance of the ToolRegistry for application-wide use.
tool_registry = ToolRegistry()