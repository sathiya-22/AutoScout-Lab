import logging
from typing import Any, Dict, Callable, Optional

# Set up a logger for this module
logger = logging.getLogger(__name__)
# Configure basic logging if not already configured by a higher-level script
# This ensures that messages are visible even when this file is run in isolation
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- MOCK TOOL DEFINITIONS ---
# In a complete project, these tool definitions and the TOOL_REGISTRY would
# typically reside in `src/tools/tool_definitions.py` and be imported.
# For the purpose of providing a self-contained implementation of
# `src/tools/tool_executor.py` as requested, a minimal set of mock tools
# and their registry are defined directly here.

def _mock_search_tool_function(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Simulates a search operation.

    Args:
        query: The search query string.
        max_results: The maximum number of results to return.

    Returns:
        A dictionary containing the search results.
    """
    logger.info(f"Mock Search Tool: Searching for '{query}' with max_results={max_results}")
    if "error" in query.lower():
        # Simulate a runtime error for specific queries
        raise ValueError("Simulated search error for queries containing 'error'")
    results = [f"Mock search result for '{query}' (item {i+1})" for i in range(max_results)]
    return {"results": results}

def _mock_calculator_tool_function(expression: str) -> Dict[str, Any]:
    """
    Simulates a calculator operation by evaluating a mathematical expression.

    WARNING: Using `eval()` with untrusted input can lead to security vulnerabilities.
    This is used here for a simple prototype demonstration of tool execution.
    In a production environment, use a safe mathematical expression parser/evaluator.

    Args:
        expression: The mathematical expression to evaluate (e.g., "2 + 2 * 3").

    Returns:
        A dictionary containing the calculated result.
    """
    logger.info(f"Mock Calculator Tool: Evaluating '{expression}'")
    try:
        # Evaluate the expression. This is for demonstration ONLY.
        result = eval(expression)
        return {"result": result}
    except Exception as e:
        # Catch any errors during evaluation (e.g., syntax errors, ZeroDivisionError)
        raise ValueError(f"Mock calculation error: Invalid expression '{expression}' - {e}")

# The default registry that maps tool names (strings) to their actual Python callable functions.
_MOCK_TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "search": _mock_search_tool_function,
    "calculator": _mock_calculator_tool_function,
}
# --- END MOCK TOOL DEFINITIONS ---


class ToolExecutor:
    """
    Executes validated tool calls.

    This class provides a safe interface to invoke tools from a predefined
    registry. It assumes that the tool calls (i.e., the `tool_name` and
    `tool_args`) have already undergone a thorough validation process
    (e.g., by Schema Validator and Hierarchical Validator). This includes
    ensuring correct tool names, argument types, adherence to schemas, and
    logical consistency with the user's intent.

    The primary role of the ToolExecutor is to:
    1. Locate the correct Python function for a given tool name.
    2. Invoke that function with the provided, validated arguments.
    3. Catch and report any runtime exceptions that occur during the tool's
       execution, providing a standardized success/failure response.
    """

    def __init__(self, tool_registry: Optional[Dict[str, Callable[..., Any]]] = None):
        """
        Initializes the ToolExecutor instance.

        Args:
            tool_registry: An optional dictionary mapping tool names (str) to
                           their corresponding Python callable functions.
                           If `None`, a default mock registry (`_MOCK_TOOL_REGISTRY`)
                           is used. In a fully built application, this would typically
                           be imported from `src/tools/tool_definitions.py`.
        """
        self._tool_registry = tool_registry if tool_registry is not None else _MOCK_TOOL_REGISTRY

        if not self._tool_registry:
            error_msg = "ToolExecutor initialized with an empty tool registry. No tools can be executed."
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"ToolExecutor initialized with {len(self._tool_registry)} tools: {list(self._tool_registry.keys())}")

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a specified tool with the given arguments.

        This method attempts to locate the tool's Python function in the
        internal registry and invoke it with the provided arguments. It wraps
        the tool's execution in a try-except block to catch and report
        any exceptions that occur during the tool's runtime.

        Args:
            tool_name: The string name of the tool to execute (e.g., "search", "calculator").
            tool_args: A dictionary where keys are argument names (str) and
                       values are their corresponding argument values (Any).
                       These arguments are expected to have been pre-validated
                       by upstream components (e.g., SchemaValidator).

        Returns:
            A dictionary containing the execution status and either the output
            of the tool or an error message:
            - On successful execution: `{"success": True, "output": <tool_function_return_value>}`
            - On failure (e.g., tool not found, runtime error in tool):
              `{"success": False, "error": <error_message>}`

        Note:
            While a check for `tool_name` existence is included, the robust
            multi-layered validation architecture (`SchemaValidator`,
            `HierarchicalValidator`) is expected to prevent invalid tool
            calls from reaching this execution stage. This check acts as
            a final safeguard.
        """
        tool_function = self._tool_registry.get(tool_name)

        if tool_function is None:
            error_msg = (
                f"Tool '{tool_name}' not found in the tool registry. "
                f"Available tools: {list(self._tool_registry.keys())}. "
                "This indicates a potential bypass of earlier validation or misconfiguration."
            )
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        logger.debug(f"Attempting to execute tool '{tool_name}' with arguments: {tool_args}")
        try:
            # Unpack the dictionary into keyword arguments for the tool function
            output = tool_function(**tool_args)
            logger.info(f"Tool '{tool_name}' executed successfully. Output type: {type(output).__name__}")
            return {"success": True, "output": output}
        except Exception as e:
            # Catch any exception raised by the tool function itself during its execution
            error_msg = (
                f"Runtime error during execution of tool '{tool_name}' "
                f"with arguments {tool_args}: {type(e).__name__}: {e}"
            )
            logger.error(error_msg, exc_info=True)  # exc_info=True logs the full traceback
            return {"success": False, "error": error_msg}