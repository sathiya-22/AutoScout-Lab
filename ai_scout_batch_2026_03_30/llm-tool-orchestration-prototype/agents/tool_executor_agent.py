```python
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError

# Project-specific imports
from config.settings import Settings
from llm_integrations.llm_client import LLMClient
from llm_integrations.prompt_templates import PromptTemplates
from llm_integrations.response_parser import ResponseParser
from tools.tool_registry import ToolRegistry
from tools.sandbox_executor import SandboxExecutor
from validation.schema_validator import SchemaValidator
from agents.verifier_agent import VerifierAgent, ProposedToolCall, VerificationDecision # Assuming these are defined in verifier_agent.py
from tools.tool_definitions import ToolDefinition # For type hints and context of tool schemas


logger = logging.getLogger(__name__)

class ToolExecutorAgent:
    """
    Dedicated agent for selecting, invoking, and managing the execution of external tools.
    Crucially, before any tool execution, it proposes the action to a VerifierAgent for explicit approval.
    """
    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        schema_validator: SchemaValidator,
        verifier_agent: VerifierAgent,
        sandbox_executor: SandboxExecutor
    ):
        self.settings = settings
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.schema_validator = schema_validator
        self.verifier_agent = verifier_agent
        self.sandbox_executor = sandbox_executor
        self.response_parser = ResponseParser() # Assumed stateless
        
        # Retrieve the specific prompt template for tool selection
        self.tool_selection_prompt_template = PromptTemplates.get_tool_selection_prompt()

        logger.info("ToolExecutorAgent initialized.")

    def _generate_tool_call_with_llm(
        self,
        task_description: str,
        current_context: Dict[str, Any],
        available_tools: List[ToolDefinition],
        previous_error: Optional[str] = None,
        retry_attempt: int = 0
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Uses the LLM to generate a tool call (name and arguments).
        Includes available tools in the prompt and handles self-correction via retry.
        Returns a tuple of (tool_name, tool_arguments) or None if unsuccessful.
        """
        if retry_attempt >= self.settings.tool_executor_max_llm_retries:
            logger.warning(
                f"Max LLM retries ({self.settings.tool_executor_max_llm_retries}) "
                f"reached for generating tool call for task: '{task_description}'."
            )
            return None

        tool_schemas_str = self.tool_registry.get_all_tool_schemas_as_json_string()
        
        error_message_for_llm = (
            f"\n\nPrevious attempt failed with error: {previous_error}. "
            "Please correct the tool call, ensuring it adheres to the schema and current context."
            if previous_error else ""
        )

        prompt = self.tool_selection_prompt_template.format(
            task_description=task_description,
            current_context=current_context,
            available_tools_json=tool_schemas_str,
            previous_error_feedback=error_message_for_llm
        )

        try:
            llm_response = self.llm_client.get_completion(
                prompt, model=self.settings.tool_executor_llm_model
            )
            parsed_call = self.response_parser.parse_tool_call_response(llm_response)
            
            if not parsed_call:
                logger.warning(f"LLM response could not be parsed. Raw response: {llm_response}")
                return self._generate_tool_call_with_llm(
                    task_description, current_context, available_tools,
                    "LLM response format was unparsable. Ensure JSON output with 'tool_name' and 'arguments'.",
                    retry_attempt + 1
                )

            tool_name = parsed_call.get("tool_name")
            tool_args = parsed_call.get("arguments")

            if not isinstance(tool_name, str) or not tool_name:
                logger.warning(f"LLM generated invalid or missing tool name: {tool_name}")
                return self._generate_tool_call_with_llm(
                    task_description, current_context, available_tools,
                    "Generated 'tool_name' is missing or not a string. Provide a valid tool name.",
                    retry_attempt + 1
                )

            if not isinstance(tool_args, dict):
                logger.warning(f"LLM generated invalid tool arguments format: {tool_args}")
                return self._generate_tool_call_with_llm(
                    task_description, current_context, available_tools,
                    "Generated 'arguments' are not in a valid dictionary format. Provide arguments as a JSON object.",
                    retry_attempt + 1
                )

            return tool_name, tool_args

        except Exception as e:
            logger.error(
                f"Error communicating with LLM or parsing response "
                f"during tool call generation: {e}", exc_info=True
            )
            return self._generate_tool_call_with_llm(
                task_description, current_context, available_tools,
                f"Internal LLM interaction error: {type(e).__name__} - {e}. Please try again.",
                retry_attempt + 1
            )

    def _validate_and_execute_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        task_description: str,
        current_context: Dict[str, Any]
    ) -> Any:
        """
        Performs schema validation, proposes the action to the VerifierAgent,
        and if approved, executes the tool via the SandboxExecutor.
        Raises ValueError, PermissionError, or RuntimeError on failure.
        """
        # 1. Schema Validation
        validation_result, validation_error = self.schema_validator.validate_tool_call(tool_name, tool_args)
        if not validation_result:
            logger.warning(
                f"Tool call schema validation failed for '{tool_name}' with args {tool_args}: "
                f"{validation_error}"
            )
            raise ValueError(f"Schema validation failed: {validation_error}")

        logger.info(f"Tool call '{tool_name}' with args {tool_args} passed initial schema validation.")

        # 2. Propose to Verifier Agent
        proposed_call = ProposedToolCall(
            tool_name=tool_name,
            arguments=tool_args,
            reasoning=(
                f"The agent proposes to call tool '{tool_name}' to accomplish the sub-task "
                f"'{task_description}'. The current state context is: {current_context}"
            )
        )
        
        verification_decision: VerificationDecision
        try:
            verification_decision = self.verifier_agent.verify_tool_call(proposed_call)
        except Exception as e:
            logger.error(
                f"VerifierAgent encountered an error during verification for '{tool_name}': {e}", 
                exc_info=True
            )
            raise RuntimeError(f"VerifierAgent failed to provide a decision: {e}")

        if not verification_decision.is_approved:
            logger.warning(
                f"Tool call '{tool_name}' with args {tool_args} was denied by VerifierAgent. "
                f"Reason: {verification_decision.reason}"
            )
            raise PermissionError(
                f"Tool execution denied by VerifierAgent: {verification_decision.reason}"
            )

        logger.info(
            f"Tool call '{tool_name}' with args {tool_args} approved by VerifierAgent. "
            f"Reason: {verification_decision.reason}"
        )

        # 3. Execute tool via SandboxExecutor
        try:
            tool_output = self.sandbox_executor.execute_tool(tool_name, tool_args)
            logger.info(f"Tool '{tool_name}' executed successfully. Output: {tool_output}")
            return tool_output
        except Exception as e:
            logger.error(
                f"Error executing tool '{tool_name}' with args {tool_args} in sandbox: {e}", 
                exc_info=True
            )
            raise RuntimeError(f"Tool execution failed in sandbox: {e}")

    def execute_tool_action(self, task_description: str, current_context: Dict[str, Any]) -> Any:
        """
        Main method for the ToolExecutorAgent to orchestrate tool selection, validation,
        verification, and execution. Implements a retry mechanism for robust operation.
        """
        logger.info(f"ToolExecutorAgent initiating tool action for task: '{task_description}'")
        logger.debug(f"Current context for tool action: {current_context}")

        available_tools = self.tool_registry.get_all_tools()
        if not available_tools:
            logger.warning("No tools registered in the ToolRegistry. Cannot execute any tool action.")
            return {"status": "error", "message": "No tools available for execution."}

        tool_call_successful = False
        tool_output = None
        current_error_message: Optional[str] = None

        # Outer retry loop for the entire tool selection, validation, and execution process
        for overall_attempt in range(self.settings.tool_executor_max_overall_retries + 1):
            try:
                # 1. Generate Tool Call with LLM, potentially self-correcting
                llm_tool_call = self._generate_tool_call_with_llm(
                    task_description,
                    current_context,
                    available_tools,
                    previous_error=current_error_message,
                    retry_attempt=0 # LLM generation has its own internal retries; reset for each overall attempt
                )

                if llm_tool_call is None:
                    logger.error(
                        f"Failed to generate a valid tool call after multiple LLM retries "
                        f"for overall attempt {overall_attempt + 1}. Aborting."
                    )
                    break # Break overall retry loop if LLM can't produce a call

                tool_name, tool_args = llm_tool_call
                logger.info(
                    f"LLM proposed tool call (Overall Attempt {overall_attempt + 1}): "
                    f"Tool='{tool_name}', Arguments={tool_args}"
                )

                # 2. Validate, Verify, and Execute the proposed tool call
                tool_output = self._validate_and_execute_tool_call(
                    tool_name, tool_args, task_description, current_context
                )
                tool_call_successful = True
                break # Exit overall retry loop on success

            except ValueError as e: # Schema validation failure or general parsing error
                current_error_message = (
                    f"Invalid tool call details: {e}. "
                    "The proposed tool call did not conform to the expected schema."
                )
                logger.warning(
                    f"Overall attempt {overall_attempt + 1} failed due to schema/parsing error: "
                    f"{current_error_message}"
                )
            except PermissionError as e: # VerifierAgent denied execution
                current_error_message = (
                    f"Tool execution denied by Verifier Agent: {e}. "
                    "The proposed action was deemed unsafe or against policy."
                )
                logger.warning(
                    f"Overall attempt {overall_attempt + 1} failed due to verifier denial: "
                    f"{current_error_message}"
                )
            except RuntimeError as e: # Sandbox execution failure or VerifierAgent internal error
                current_error_message = (
                    f"Execution or verification system error: {e}. "
                    "Please check tool implementation, sandbox environment, or verifier agent status."
                )
                logger.error(
                    f"Overall attempt {overall_attempt + 1} failed during execution/verification: "
                    f"{current_error_message}", exc_info=True
                )
            except Exception as e:
                current_error_message = (
                    f"An unexpected internal error occurred during tool action: {type(e).__name__} - {e}. "
                    "Please review system logs for details."
                )
                logger.critical(
                    f"Critical unexpected error in tool execution loop (Overall Attempt {overall_attempt + 1}): {e}", 
                    exc_info=True
                )
            
            if overall_attempt == self.settings.tool_executor_max_overall_retries:
                logger.error(
                    f"Failed to execute tool action after {self.settings.tool_executor_max_overall_retries + 1} "
                    f"overall attempts for task: '{task_description}'."
                )

        if tool_call_successful:
            logger.info(f"Tool action completed successfully for task: '{task_description}'.")
            return {"status": "success", "output": tool_output}
        else:
            final_error_message = current_error_message or "Failed to execute tool action after multiple attempts due to an unknown error."
            logger.error(f"Tool action failed for task: '{task_description}'. Reason: {final_error_message}")
            return {"status": "error", "message": final_error_message}

```