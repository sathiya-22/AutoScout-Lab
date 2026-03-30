from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class BaseToolInput(BaseModel):
    """Base class for all tool input schemas."""
    pass

class BaseToolOutput(BaseModel):
    """Base class for all tool output schemas."""
    success: bool = Field(..., description="True if the tool execution was successful, False otherwise.")
    error_message: Optional[str] = Field(None, description="Error message if the tool execution failed.")

class ToolDefinition:
    """
    A class to encapsulate the definition of a tool, including its name, description,
    and Pydantic schemas for its input arguments and output.
    """
    def __init__(self, name: str, description: str, input_schema: type[BaseToolInput], output_schema: type[BaseToolOutput]):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema

    def to_json_schema(self) -> Dict[str, Any]:
        """Generates a JSON schema representation of the tool for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema.model_json_schema(),
        }

# --- Example Tool Definitions ---

# 1. Search Tool
class SearchToolInput(BaseToolInput):
    query: str = Field(..., description="The search query string.")
    num_results: int = Field(5, description="Number of search results to retrieve.", ge=1, le=10)

class SearchToolResult(BaseModel):
    title: str = Field(..., description="Title of the search result.")
    url: str = Field(..., description="URL of the search result.")
    snippet: str = Field(..., description="Brief snippet of the search result content.")

class SearchToolOutput(BaseToolOutput):
    results: List[SearchToolResult] = Field(..., description="List of search results.")

SearchTool = ToolDefinition(
    name="search_web",
    description="Searches the web for information based on a query. Use this for general knowledge or up-to-date information.",
    input_schema=SearchToolInput,
    output_schema=SearchToolOutput
)

# 2. Calculator Tool
class CalculatorToolInput(BaseToolInput):
    expression: str = Field(..., description="The mathematical expression to evaluate, e.g., '2 + 2 * (3 / 1.5)'.")

class CalculatorToolOutput(BaseToolOutput):
    result: float = Field(..., description="The numerical result of the evaluated expression.")
    
CalculatorTool = ToolDefinition(
    name="calculate",
    description="Evaluates a mathematical expression and returns the result. Supports basic arithmetic operations.",
    input_schema=CalculatorToolInput,
    output_schema=CalculatorToolOutput
)

# 3. File Read Tool
class FileReadToolInput(BaseToolInput):
    file_path: str = Field(..., description="The path to the file to read.")

class FileReadToolOutput(BaseToolOutput):
    content: str = Field(..., description="The content of the file.")

FileReadTool = ToolDefinition(
    name="read_file",
    description="Reads the content of a specified file.",
    input_schema=FileReadToolInput,
    output_schema=FileReadToolOutput
)

# 4. JSON Editor Tool
class JsonEditorToolInput(BaseToolInput):
    json_data: Dict[str, Any] = Field(..., description="The JSON object to modify or analyze.")
    path: str = Field(..., description="JSONPath expression to target a specific element, e.g., '$.users[0].name'.")
    new_value: Optional[Any] = Field(None, description="The new value to set at the specified JSONPath. If None, it implies reading the value.")

class JsonEditorToolOutput(BaseToolOutput):
    modified_json: Optional[Dict[str, Any]] = Field(None, description="The modified JSON object if a new value was provided.")
    extracted_value: Optional[Any] = Field(None, description="The value extracted at the specified path if no new value was provided.")

JsonEditorTool = ToolDefinition(
    name="json_editor",
    description="A tool to read, modify, or analyze JSON objects using JSONPath expressions.",
    input_schema=JsonEditorToolInput,
    output_schema=JsonEditorToolOutput
)

# A list of all defined tools for easy registration
ALL_TOOLS: List[ToolDefinition] = [
    SearchTool,
    CalculatorTool,
    FileReadTool,
    JsonEditorTool
]