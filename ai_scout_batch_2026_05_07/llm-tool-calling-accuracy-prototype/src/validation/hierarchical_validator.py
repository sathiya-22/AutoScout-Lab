from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

# Assume these models and classes exist in their respective paths
from src.tools.tool_definitions import ToolDefinition
from src.validation.schema_validator import SchemaValidator, SchemaValidationResult
from src.agents.validation_agent import ValidationAgent, ValidationAgentResult
from src.utils.data_models import ToolCall # Assuming ToolCall is the standard validated object

class HierarchicalValidationResult(BaseModel):
    """
    Represents the comprehensive outcome of the hierarchical validation process.
    """
    is_valid: bool
    final_tool_call: Optional[ToolCall] = None
    error_message: Optional[str] = None
    feedback_for_primary_agent: Optional[str] = None
    validation_steps_taken: List[str] = Field(default_factory=list)

class HierarchicalValidator:
    """
    Orchestrates the multi-layered validation process for proposed tool calls.
    It first uses a SchemaValidator for deterministic checks, and optionally
    involves a ValidationAgent for higher-level contextual and semantic review
    or error correction.
    """

    def __init__(self):
        self._schema_validator = SchemaValidator()
        self._validation_agent = ValidationAgent()

    async def validate(
        self,
        user_query: str,
        primary_agent_output: str, # Raw text output from the primary agent
        available_tools: List[ToolDefinition],
        require_validation_agent_for_valid_schema: bool = True
    ) -> HierarchicalValidationResult:
        """
        Performs hierarchical validation on a proposed tool call.

        Args:
            user_query: The original user request.
            primary_agent_output: The raw text output from the primary agent, expected
                                  to contain a tool call or indicate no tool call.
            available_tools: A list of ToolDefinition objects the agent can use.
            require_validation_agent_for_valid_schema: If True, the ValidationAgent will
                                                       always be involved even if schema
                                                       validation passes, for deeper
                                                       contextual/semantic checks.
                                                       If False, the ValidationAgent
                                                       is only invoked if schema
                                                       validation fails to attempt correction.

        Returns:
            A HierarchicalValidationResult indicating validity and the final
            (potentially corrected) ToolCall object.
        """
        steps_taken = []

        # 1. Initial Schema Validation of the primary agent's raw output
        schema_validation_result = self._schema_validator.validate_primary_agent_output(
            primary_agent_output=primary_agent_output,
            available_tools=available_tools
        )
        steps_taken.append(f"Schema Validation: {'Passed' if schema_validation_result.is_valid else 'Failed'}")

        if schema_validation_result.is_valid:
            # Schema validation passed.
            if not require_validation_agent_for_valid_schema:
                # If no further agent validation is required for a schema-valid call, we're done.
                return HierarchicalValidationResult(
                    is_valid=True,
                    final_tool_call=schema_validation_result.parsed_tool_call,
                    validation_steps_taken=steps_taken
                )
            else:
                # Schema passed, but Validation Agent is required for deeper checks
                # (e.g., contextual, logical consistency).
                steps_taken.append("Involving Validation Agent for deeper contextual/semantic check.")
                agent_validation_result = await self._validation_agent.validate_and_correct(
                    user_query=user_query,
                    proposed_tool_call=schema_validation_result.parsed_tool_call,
                    primary_agent_raw_output=primary_agent_output, # Provide raw output for context
                    available_tools=available_tools
                )
                steps_taken.append(f"Validation Agent Review: {'Approved' if agent_validation_result.is_valid else 'Rejected/Corrected'}")

                return HierarchicalValidationResult(
                    is_valid=agent_validation_result.is_valid,
                    final_tool_call=agent_validation_result.corrected_tool_call or schema_validation_result.parsed_tool_call,
                    error_message=agent_validation_result.error_message,
                    feedback_for_primary_agent=agent_validation_result.feedback_for_primary_agent,
                    validation_steps_taken=steps_taken
                )
        else:
            # Schema validation failed. Involve Validation Agent to attempt correction
            # or provide detailed feedback.
            steps_taken.append("Schema validation failed, involving Validation Agent for correction or detailed feedback.")
            # The agent gets the *raw* output and the schema error to help it diagnose.
            agent_validation_result = await self._validation_agent.validate_and_correct(
                user_query=user_query,
                proposed_tool_call=None, # No valid parsed call from schema validator
                primary_agent_raw_output=primary_agent_output,
                schema_validation_error=schema_validation_result.error_message,
                available_tools=available_tools
            )
            steps_taken.append(f"Validation Agent Correction Attempt: {'Successful' if agent_validation_result.is_valid else 'Failed'}")

            # If the validation agent managed to propose a correction, re-validate it against the schema.
            if agent_validation_result.is_valid and agent_validation_result.corrected_tool_call:
                re_schema_validation_result = self._schema_validator.validate_tool_call_object(
                    tool_call=agent_validation_result.corrected_tool_call,
                    available_tools=available_tools
                )
                steps_taken.append(f"Re-Schema Validation of Agent's Correction: {'Passed' if re_schema_validation_result.is_valid else 'Failed'}")

                if re_schema_validation_result.is_valid:
                    return HierarchicalValidationResult(
                        is_valid=True,
                        final_tool_call=re_schema_validation_result.parsed_tool_call,
                        validation_steps_taken=steps_taken
                    )
                else:
                    # Agent's proposed correction was still schema-invalid.
                    full_error = (
                        f"Validation agent proposed a correction, but it failed re-schema validation: "
                        f"{re_schema_validation_result.error_message}"
                    )
                    full_feedback = (
                        f"{agent_validation_result.feedback_for_primary_agent or ''}\n"
                        f"Correction failed schema re-validation: {re_schema_validation_result.error_message}"
                    ).strip()
                    return HierarchicalValidationResult(
                        is_valid=False,
                        final_tool_call=None,
                        error_message=full_error,
                        feedback_for_primary_agent=full_feedback,
                        validation_steps_taken=steps_taken
                    )
            else:
                # Validation agent could not correct, or didn't propose one.
                return HierarchicalValidationResult(
                    is_valid=False,
                    final_tool_call=None,
                    error_message=agent_validation_result.error_message or "Schema validation failed and validation agent could not correct it.",
                    feedback_for_primary_agent=agent_validation_result.feedback_for_primary_agent,
                    validation_steps_taken=steps_taken
                )