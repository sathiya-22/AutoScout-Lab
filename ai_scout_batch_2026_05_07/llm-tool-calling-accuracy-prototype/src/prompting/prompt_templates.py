```python
import json
from typing import List, Dict, Any, Optional

class PromptTemplates:
    """
    Manages and generates advanced prompt templates for the Primary Agent.
    These templates incorporate system instructions, available tool definitions,
    few-shot examples, Chain-of-Thought reasoning instructions, and explicit
    structured output formats to maximize tool calling accuracy.
    """

    SYSTEM_MESSAGE = (
        "You are an intelligent AI assistant whose primary goal is to accurately identify and execute tools "
        "based on user requests. Your responses must strictly adhere to the specified JSON format for tool calls. "
        "Before generating the final JSON, you MUST think step-by-step to ensure correctness and adherence "
        "to tool schemas."
    )

    TOOL_DEFINITIONS_PREFIX = "\n\n# Available Tools\n"

    FEW_SHOT_PREFIX = "\n\n# Examples of Correct Tool Usage\n"
    FEW_SHOT_EXAMPLE_FORMAT = (
        "--- Example Start ---\n"
        "User Request: {user_input}\n"
        "Thought: {thought}\n"
        "Tool Call:\n```json\n{tool_call_json}\n```\n"
        "--- Example End ---\n"
    )

    CHAIN_OF_THOUGHT_INSTRUCTIONS = (
        "\n\n# Reasoning Process\n"
        "Before making a tool call, carefully follow these steps to determine the correct tool and arguments:\n"
        "1.  **Understand Intent**: Analyze the 'User Request' to grasp the user's core intent and any implied actions.\n"
        "2.  **Tool Selection**: Review the 'Available Tools' and their descriptions. Select the tool that most accurately "
        "    matches the user's intent. If no tool is suitable, clearly state that you cannot fulfill the request.\n"
        "3.  **Parameter Extraction**: For the chosen tool, identify all required and relevant optional parameters. "
        "    Extract the necessary values directly from the 'User Request'. Pay close attention to data types "
        "    (e.g., numbers, strings, booleans, specific formats).\n"
        "4.  **Validation & Defaulting**: Check if all *required* parameters have been found. If an optional parameter "
        "    is not provided in the request, consider if a sensible default can be used or if it should be omitted. "
        "    Do not invent values. If critical information is missing, state this in your thought process.\n"
        "5.  **Construct Tool Call**: Assemble the tool call in the specified JSON format, ensuring the tool name is exact "
        "    and all arguments are correctly named, typed, and populated.\n"
        "6.  **Self-Correction**: Briefly review your chosen tool and arguments. Does it perfectly address the user's request? "
        "    Are there any ambiguities or potential misinterpretations? Correct any errors you find.\n"
        "Now, output your step-by-step reasoning under the heading 'Thought:'.\n"
    )

    OUTPUT_FORMAT_INSTRUCTIONS = (
        "\n\n# Output Format\n"
        "After your 'Thought:' section, provide the final tool call in a JSON object. "
        "The JSON must strictly contain two top-level keys: 'tool_name' (string) and 'arguments' (JSON object). "
        "Ensure all required arguments for the selected tool are present and correctly typed according to its schema. "
        "If no tool call is appropriate, you may output a simple JSON indicating that, e.g., "
        "`{{\"tool_name\": null, \"arguments\": null, \"reason\": \"No suitable tool found.\"}}`.\n"
        "```json\n"
        "{{\n"
        "  \"tool_name\": \"<THE_SELECTED_TOOL_NAME>\",\n"
        "  \"arguments\": {{\n"
        "    \"<PARAMETER_NAME_1>\": <VALUE_1_APPROPRIATE_TYPE>,\n"
        "    \"<PARAMETER_NAME_2>\": <VALUE_2_APPROPRIATE_TYPE>\n"
        "    // ... other parameters\n"
        "  }}\n"
        "}}\n"
        "```"
    )

    FINAL_USER_REQUEST_PREFIX = "\n\n# User Request\n"


    @staticmethod
    def _format_tool_schema(tool: Dict[str, Any]) -> str:
        """
        Formats a single tool schema dictionary into a human-readable string
        for inclusion in the prompt.
        """
        tool_name = tool.get('name', 'UNKNOWN_TOOL')
        tool_description = tool.get('description', 'No description provided.')
        parameters = tool.get('parameters', [])

        param_strings = []
        if parameters:
            param_strings.append("Parameters:")
            for param in parameters:
                param_name = param.get('name', 'UNKNOWN_PARAM')
                param_type = param.get('type', 'any')
                param_description = param.get('description', 'No description.')
                param_required = "(Required)" if param.get('required', False) else "(Optional)"
                param_strings.append(
                    f"  - {param_name} ({param_type}) {param_required}: {param_description}"
                )
        else:
            param_strings.append("Parameters: None")

        return (
            f"Tool: {tool_name}\n"
            f"Description: {tool_description}\n"
            f"{'\n'.join(param_strings)}\n"
        )

    @staticmethod
    def _format_few_shot_example(example: Dict[str, Any]) -> str:
        """
        Formats a single few-shot example dictionary into a string
        for inclusion in the prompt.
        Assumes 'tool_call_json' is either a dict or a valid JSON string.
        """
        user_input = example.get('user_input', 'No user input provided for this example.')
        thought = example.get('thought', 'No detailed thought process provided for this example.')
        tool_call = example.get('tool_call_json', {})
        
        # Ensure tool_call is a string, assuming it's already JSON or converting from dict
        if isinstance(tool_call, dict):
            tool_call_str = json.dumps(tool_call, indent=2)
        elif isinstance(tool_call, str):
            tool_call_str = tool_call # Assume it's already a valid JSON string
        else:
            tool_call_str = json.dumps(
                {"error": "Invalid tool call format in few-shot example; must be dict or JSON string."}, 
                indent=2
            )

        return PromptTemplates.FEW_SHOT_EXAMPLE_FORMAT.format(
            user_input=user_input,
            thought=thought,
            tool_call_json=tool_call_str
        )

    @staticmethod
    def get_main_prompt(
        tool_schemas: List[Dict[str, Any]],
        user_query: str,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Assembles the complete prompt for the Primary Agent, integrating system instructions,
        available tool definitions, few-shot examples, Chain-of-Thought guidance,
        and explicit output format instructions.

        Args:
            tool_schemas: A list of dictionaries, where each dictionary represents
                          a tool's schema. Expected keys for each tool:
                          'name' (str), 'description' (str), 'parameters' (list of dicts).
                          Each parameter dict expected to have 'name' (str), 'type' (str),
                          'description' (str), 'required' (bool).
            user_query: The current user request string that the agent needs to process.
            few_shot_examples: An optional list of dictionaries, each containing
                               'user_input' (str), 'thought' (str), and 'tool_call_json' (dict or JSON string)
                               to demonstrate correct tool usage.

        Returns:
            The complete, formatted prompt string ready to be sent to an LLM.
        """
        prompt_parts = [PromptTemplates.SYSTEM_MESSAGE]

        # Add tool definitions section
        prompt_parts.append(PromptTemplates.TOOL_DEFINITIONS_PREFIX)
        if tool_schemas:
            for tool in tool_schemas:
                prompt_parts.append(PromptTemplates._format_tool_schema(tool))
        else:
            prompt_parts.append("No tools are currently defined or available for use.")

        # Add few-shot examples section if provided
        if few_shot_examples:
            prompt_parts.append(PromptTemplates.FEW_SHOT_PREFIX)
            for example in few_shot_examples:
                prompt_parts.append(PromptTemplates._format_few_shot_example(example))

        # Add Chain-of-Thought instructions
        prompt_parts.append(PromptTemplates.CHAIN_OF_THOUGHT_INSTRUCTIONS)

        # Add output format instructions
        prompt_parts.append(PromptTemplates.OUTPUT_FORMAT_INSTRUCTIONS)

        # Add the current user request
        prompt_parts.append(PromptTemplates.FINAL_USER_REQUEST_PREFIX)
        prompt_parts.append(user_query)
        
        # Conclude with a prompt for the 'Thought:' section to kick off CoT
        prompt_parts.append("\nThought:")

        return "\n".join(prompt_parts)

```