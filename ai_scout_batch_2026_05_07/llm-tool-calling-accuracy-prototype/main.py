```python
import logging
from typing import Any, Dict, Optional, List

# Placeholder for Pydantic if not explicitly used for ALL internal schemas
try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        pass
    class Field:
        def __init__(self, *args, **kwargs):
            pass

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Placeholder Classes for Architecture Components ---
# These classes simulate the interfaces of the actual components
# without implementing their full internal logic.

class ToolCall(BaseModel):
    """Represents a proposed tool call."""
    tool_name: str
    args: Dict[str, Any]

class ValidationResult(BaseModel):
    """Result of the validation process."""
    is_valid: bool
    error_message: Optional[str] = None
    corrected_call: Optional[ToolCall] = None # If validation agent directly corrected

class CorrectionOutcome(BaseModel):
    """Outcome of attempting to correct a tool call."""
    was_corrected: bool
    needs_reprompt: bool = False # If Primary Agent needs to be re-prompted
    corrected_call: Optional[ToolCall] = None # The directly corrected call
    feedback: Optional[str] = None # Feedback for the Primary Agent if re-prompting

class PrimaryAgent:
    """
    Simulates the Primary Agent from src/agents/primary_agent.py.
    Interprets user requests and proposes tool calls.
    """
    def __init__(self, llm_config: Dict[str, Any]):
        logger.info("PrimaryAgent initialized.")
        self.llm_config = llm_config # Simulate LLM configuration

    def propose_tool_call(self, user_query: str, available_tools_schema: Dict[str, Any]) -> Optional[ToolCall]:
        """
        Proposes a tool call based on the user query and available tool schemas.
        In a real implementation, this would involve LLM inference with advanced prompting.
        """
        logger.info(f"PrimaryAgent proposing tool call for query: '{user_query}'")
        # Simulate LLM output based on query
        if "weather in London" in user_query.lower():
            return ToolCall(tool_name="get_current_weather", args={"location": "London", "unit": "celsius"})
        elif "schedule a meeting" in user_query.lower():
            # Intentionally return an invalid call for demonstration of validation
            return ToolCall(tool_name="schedule_event", args={"title": "Team Sync", "time": "tomorrow 10am"}) # Missing 'date'
        elif "email" in user_query.lower() and "john" in user_query.lower():
            return ToolCall(tool_name="send_email", args={"recipient": "john.doe@example.com", "subject": "Quick Catch-up", "body": "Hi John, let's connect."})
        elif "invalid_tool_call" in user_query.lower():
            return ToolCall(tool_name="non_existent_tool", args={"param": "value"})
        else:
            logger.warning("PrimaryAgent could not determine a tool call.")
            return None

    def re_prompt(self, user_query: str, available_tools_schema: Dict[str, Any], feedback: str) -> Optional[ToolCall]:
        """
        Re-prompts the primary agent with specific feedback for self-correction.
        """
        logger.info(f"PrimaryAgent re-prompted with feedback: '{feedback}' for query: '{user_query}'")
        # In a real scenario, this would involve another LLM call with the feedback
        # For simulation, let's assume it fixes a known error based on feedback
        if "missing 'date' for 'schedule_event'" in feedback:
            logger.info("PrimaryAgent self-corrected 'schedule_event' call.")
            return ToolCall(tool_name="schedule_event", args={"title": "Team Sync", "time": "10:00", "date": "2023-11-20"}) # Corrected
        elif "invalid unit" in feedback:
            logger.info("PrimaryAgent self-corrected 'get_current_weather' unit.")
            return ToolCall(tool_name="get_current_weather", args={"location": "Paris", "unit": "fahrenheit"})
        
        logger.warning(f"PrimaryAgent failed to self-correct based on feedback: {feedback}")
        return None


class SchemaValidator:
    """
    Simulates the Schema Validator from src/validation/schema_validator.py.
    Performs deterministic checks against tool schemas.
    """
    def validate(self, tool_call: ToolCall, tool_schemas: Dict[str, Any]) -> ValidationResult:
        """
        Validates a tool call against its defined schema.
        """
        logger.debug(f"SchemaValidator validating call: {tool_call.tool_name} with args {tool_call.args}")
        tool_name = tool_call.tool_name
        args = tool_call.args

        if tool_name not in tool_schemas:
            return ValidationResult(is_valid=False, error_message=f"Unknown tool: '{tool_name}'")

        schema = tool_schemas[tool_name]
        required_params = schema.get("parameters", {}).get("required", [])
        properties = schema.get("parameters", {}).get("properties", {})

        # Check for missing required parameters
        for param in required_params:
            if param not in args:
                return ValidationResult(is_valid=False, error_message=f"Missing required parameter: '{param}' for tool '{tool_name}'")

        # Check argument types and constraints (simplified)
        for arg_name, arg_value in args.items():
            if arg_name not in properties:
                # Allow extra parameters for now, or flag as error depending on strictness
                logger.warning(f"Parameter '{arg_name}' not defined in schema for tool '{tool_name}'.")
                continue

            param_schema = properties[arg_name]
            expected_type = param_schema.get("type")
            
            # Basic type checking
            if expected_type == "string" and not isinstance(arg_value, str):
                return ValidationResult(is_valid=False, error_message=f"Parameter '{arg_name}' expects string, got {type(arg_value).__name__}")
            elif expected_type == "integer" and not isinstance(arg_value, int):
                # Try to cast if possible, for robustness
                try:
                    args[arg_name] = int(arg_value)
                except (ValueError, TypeError):
                    return ValidationResult(is_valid=False, error_message=f"Parameter '{arg_name}' expects integer, got {type(arg_value).__name__}")
            elif expected_type == "number" and not isinstance(arg_value, (int, float)):
                 try:
                    args[arg_name] = float(arg_value)
                 except (ValueError, TypeError):
                    return ValidationResult(is_valid=False, error_message=f"Parameter '{arg_name}' expects number, got {type(arg_value).__name__}")
            
            # Check enums
            enum_values = param_schema.get("enum")
            if enum_values and arg_value not in enum_values:
                return ValidationResult(is_valid=False, error_message=f"Parameter '{arg_name}' has invalid value '{arg_value}'. Must be one of {enum_values}")
            
            # Additional checks for specific tools (e.g., date format for schedule_event)
            if tool_name == "schedule_event":
                if arg_name == "date":
                    # Simple date format check (e.g., YYYY-MM-DD)
                    if not isinstance(arg_value, str) or len(arg_value) != 10 or arg_value[4] != '-' or arg_value[7] != '-':
                        return ValidationResult(is_valid=False, error_message=f"Parameter 'date' for 'schedule_event' has invalid format. Expected YYYY-MM-DD, got '{arg_value}'")


        return ValidationResult(is_valid=True)

class ValidationAgent:
    """
    Simulates the Validation Agent from src/agents/validation_agent.py.
    An LLM agent for reviewing and potentially correcting tool calls.
    """
    def __init__(self, llm_config: Dict[str, Any]):
        logger.info("ValidationAgent initialized.")
        self.llm_config = llm_config

    def review_and_correct(self, user_query: str, proposed_tool_call: ToolCall,
                           tool_schemas: Dict[str, Any], schema_validation_feedback: Optional[str]) -> ValidationResult:
        """
        Reviews the proposed tool call, considering user query, tool schemas,
        and schema validation feedback. Can directly correct the call.
        """
        logger.info(f"ValidationAgent reviewing tool call: {proposed_tool_call.tool_name} for query: '{user_query}'")
        logger.debug(f"Schema validation feedback: {schema_validation_feedback}")

        # Simulate LLM logic to review and correct
        if schema_validation_feedback and "Missing required parameter: 'date' for tool 'schedule_event'" in schema_validation_feedback:
            logger.info("ValidationAgent correcting missing 'date' for schedule_event.")
            corrected_call = ToolCall(
                tool_name=proposed_tool_call.tool_name,
                args={**proposed_tool_call.args, "date": "2023-11-20"} # Assuming a default or inferring a generic date
            )
            return ValidationResult(is_valid=True, corrected_call=corrected_call, error_message="Validation agent added default date.")
        
        if schema_validation_feedback and "invalid value 'celsius' for parameter 'unit'" in schema_validation_feedback:
            logger.info("ValidationAgent correcting invalid 'unit' for get_current_weather.")
            corrected_call = ToolCall(
                tool_name=proposed_tool_call.tool_name,
                args={**proposed_tool_call.args, "unit": "metric"} # Correcting to a valid enum value
            )
            return ValidationResult(is_valid=True, corrected_call=corrected_call, error_message="Validation agent corrected unit parameter.")

        # If schema validation passed but there's a logical issue (e.g., unit for weather)
        if proposed_tool_call.tool_name == "get_current_weather" and proposed_tool_call.args.get("unit") == "celsius":
            # This logic would be from an LLM that understands units
            if "London" in user_query and "celsius" in user_query:
                logger.info("ValidationAgent found unit 'celsius' for London, which is valid.")
                return ValidationResult(is_valid=True)
            else:
                logger.warning("ValidationAgent suspects 'celsius' might not be the default/expected unit for general weather. Suggesting a change.")
                # This could be a case where the validation agent suggests a change even if schema is valid
                corrected_call = ToolCall(tool_name="get_current_weather", args={"location": proposed_tool_call.args["location"], "unit": "metric"})
                return ValidationResult(is_valid=True, corrected_call=corrected_call, error_message="Validation agent preferred 'metric' unit.")


        # If no specific correction logic applies, defer to schema validator result
        if schema_validation_feedback:
             return ValidationResult(is_valid=False, error_message=f"Validation Agent could not resolve: {schema_validation_feedback}")

        return ValidationResult(is_valid=True) # Assume valid if no issues found by validation agent

class HierarchicalValidator:
    """
    Simulates the Hierarchical Validator from src/validation/hierarchical_validator.py.
    Orchestrates schema and LLM-based validation.
    """
    def __init__(self, schema_validator: SchemaValidator, validation_agent: ValidationAgent):
        logger.info("HierarchicalValidator initialized.")
        self.schema_validator = schema_validator
        self.validation_agent = validation_agent

    def validate(self, user_query: str, proposed_tool_call: ToolCall, tool_schemas: Dict[str, Any]) -> ValidationResult:
        """
        Runs a multi-layered validation process.
        """
        logger.info("HierarchicalValidator: Starting validation pipeline.")

        if not proposed_tool_call:
            return ValidationResult(is_valid=False, error_message="No tool call proposed by Primary Agent.")

        # 1. Schema Validation
        schema_result = self.schema_validator.validate(proposed_tool_call, tool_schemas)
        if not schema_result.is_valid:
            logger.warning(f"HierarchicalValidator: Schema validation failed: {schema_result.error_message}. Escalating to Validation Agent.")
            # Even if schema fails, the validation agent might be able to correct it or provide better feedback
            validation_agent_result = self.validation_agent.review_and_correct(
                user_query, proposed_tool_call, tool_schemas, schema_result.error_message
            )
            if validation_agent_result.is_valid:
                logger.info("HierarchicalValidator: Validation Agent successfully corrected schema error.")
                return validation_agent_result
            else:
                logger.error(f"HierarchicalValidator: Validation Agent could not fix schema error: {validation_agent_result.error_message}")
                return validation_agent_result # Pass on the agent's failure message
        else:
            logger.info("HierarchicalValidator: Schema validation passed. Proceeding to Validation Agent for logical review.")
            # 2. Validation Agent (Meta-agent) Review
            validation_agent_result = self.validation_agent.review_and_correct(
                user_query, proposed_tool_call, tool_schemas, None # No schema error to report
            )
            if not validation_agent_result.is_valid:
                logger.error(f"HierarchicalValidator: Validation Agent found logical error: {validation_agent_result.error_message}")
            elif validation_agent_result.corrected_call:
                logger.info("HierarchicalValidator: Validation Agent made a correction (e.g., improved parameters).")
            else:
                logger.info("HierarchicalValidator: Validation Agent found no issues.")
            
            return validation_agent_result

class CorrectionMechanisms:
    """
    Simulates the Correction Mechanisms from src/validation/correction_mechanisms.py.
    Applies strategies to rectify errors based on validation outcomes.
    """
    def __init__(self):
        logger.info("CorrectionMechanisms initialized.")

    def apply_correction(self, user_query: str, proposed_tool_call: ToolCall,
                         validation_result: ValidationResult, tool_schemas: Dict[str, Any]) -> CorrectionOutcome:
        """
        Applies correction strategies based on the validation result.
        """
        logger.info(f"CorrectionMechanisms attempting to apply correction for: {validation_result.error_message}")

        if not validation_result.is_valid:
            # Case 1: Validation Agent already provided a corrected call
            if validation_result.corrected_call:
                logger.info("CorrectionMechanisms: Using corrected call provided by Validation Agent.")
                return CorrectionOutcome(was_corrected=True, corrected_call=validation_result.corrected_call)

            # Case 2: Deterministic corrections based on error message
            if "Missing required parameter: 'time' for tool 'schedule_event'" in validation_result.error_message:
                logger.info("CorrectionMechanisms: Deterministically adding default 'time' to schedule_event.")
                corrected_args = {**proposed_tool_call.args, "time": "12:00"}
                return CorrectionOutcome(was_corrected=True, corrected_call=ToolCall(tool_name=proposed_tool_call.tool_name, args=corrected_args))
            
            if "expects integer, got str" in validation_result.error_message:
                logger.info("CorrectionMechanisms: Attempting to cast string to integer.")
                # This is a generic attempt; real implementation needs to parse error message for specific param
                try:
                    # Find the parameter that caused the error and try to cast it
                    error_param_match = next((k for k, v in proposed_tool_call.args.items() if isinstance(v, str) and v.isdigit()), None)
                    if error_param_match:
                        corrected_args = {**proposed_tool_call.args, error_param_match: int(proposed_tool_call.args[error_param_match])}
                        return CorrectionOutcome(was_corrected=True, corrected_call=ToolCall(tool_name=proposed_tool_call.tool_name, args=corrected_args))
                except ValueError:
                    pass # Failed to cast

            # Case 3: Provide feedback for Primary Agent to re-prompt
            if "Unknown tool" in validation_result.error_message or "Missing required parameter" in validation_result.error_message:
                logger.info("CorrectionMechanisms: Providing feedback to Primary Agent for re-prompt.")
                feedback = f"Original proposal was invalid: {validation_result.error_message}. Please review and try again."
                return CorrectionOutcome(was_corrected=False, needs_reprompt=True, feedback=feedback)

        logger.warning(f"CorrectionMechanisms: No correction applied for: {validation_result.error_message}")
        return CorrectionOutcome(was_corrected=False)

class ToolExecutor:
    """
    Simulates the Tool Executor from src/tools/tool_executor.py.
    Handles safe invocation of validated tool calls.
    """
    def __init__(self):
        logger.info("ToolExecutor initialized.")

    def execute_tool_call(self, tool_call: ToolCall) -> Any:
        """
        Executes the given tool call.
        """
        logger.info(f"Executing tool: {tool_call.tool_name} with arguments: {tool_call.args}")
        try:
            # Simulate tool execution results
            if tool_call.tool_name == "get_current_weather":
                location = tool_call.args.get("location", "Unknown")
                unit = tool_call.args.get("unit", "celsius")
                return f"The current weather in {location} is 15 degrees {unit} and partly cloudy."
            elif tool_call.tool_name == "schedule_event":
                title = tool_call.args.get("title", "Event")
                date = tool_call.args.get("date", "Unknown Date")
                time = tool_call.args.get("time", "Unknown Time")
                return f"Event '{title}' scheduled successfully for {date} at {time}."
            elif tool_call.tool_name == "send_email":
                recipient = tool_call.args.get("recipient", "unknown")
                subject = tool_call.args.get("subject", "No Subject")
                return f"Email with subject '{subject}' sent to {recipient}."
            else:
                logger.error(f"Attempted to execute unknown or unimplemented tool: {tool_call.tool_name}")
                return {"error": f"Tool '{tool_call.tool_name}' not implemented for execution."}
        except Exception as e:
            logger.exception(f"Error during tool execution for {tool_call.tool_name}: {e}")
            return {"error": f"Execution failed: {e}"}

# --- Tool Definitions (src/tools/tool_definitions.py) ---
# Using a dictionary to simulate JSON schema definitions
AVAILABLE_TOOLS_SCHEMA: Dict[str, Any] = {
    "get_current_weather": {
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit", "metric"], "description": "The unit of temperature to use. Defaults to celsius"},
            },
            "required": ["location"],
        },
    },
    "schedule_event": {
        "description": "Schedule a new event or meeting in the calendar",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title of the event"},
                "date": {"type": "string", "description": "Date of the event in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Time of the event in HH:MM format"},
                "duration_minutes": {"type": "integer", "description": "Duration of the event in minutes"},
                "attendees": {"type": "array", "items": {"type": "string", "format": "email"}, "description": "List of attendee emails"},
            },
            "required": ["title", "date", "time"],
        },
    },
    "send_email": {
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "format": "email", "description": "The email address of the recipient"},
                "subject": {"type": "string", "description": "The subject of the email"},
                "body": {"type": "string", "description": "The body content of the email"},
            },
            "required": ["recipient", "subject", "body"],
        },
    }
}

# --- Main Orchestration Logic (main.py) ---

def run_tool_calling_pipeline(user_query: str, max_retries: int = 3) -> Optional[Any]:
    """
    Drives the entire tool calling flow:
    User query -> Primary Agent -> Hierarchical Validation -> Correction/Re-prompt loop -> Tool Execution.
    """
    logger.info(f"\n--- Starting pipeline for query: '{user_query}' ---")

    # Initialize components
    # Using dummy LLM config for placeholders
    llm_config = {"model": "gpt-4-turbo", "temperature": 0.7} 
    primary_agent = PrimaryAgent(llm_config)
    schema_validator = SchemaValidator()
    validation_agent = ValidationAgent(llm_config)
    hierarchical_validator = HierarchicalValidator(schema_validator, validation_agent)
    correction_mechanisms = CorrectionMechanisms()
    tool_executor = ToolExecutor()

    current_proposed_tool_call: Optional[ToolCall] = None
    final_validated_tool_call: Optional[ToolCall] = None
    
    # --- Step 1: Primary Agent Proposes Initial Tool Call ---
    current_proposed_tool_call = primary_agent.propose_tool_call(user_query, AVAILABLE_TOOLS_SCHEMA)

    if not current_proposed_tool_call:
        logger.error("Initial proposal from Primary Agent was empty. Cannot proceed.")
        return None
    
    logger.info(f"Primary Agent initial proposal: {current_proposed_tool_call.tool_name}({current_proposed_tool_call.args})")

    # --- Step 2: Validation and Correction Loop ---
    for retry_count in range(max_retries):
        logger.info(f"\n--- Validation Attempt {retry_count + 1}/{max_retries} ---")
        
        if not current_proposed_tool_call:
            logger.error("No proposed tool call available for validation. Exiting loop.")
            break

        validation_result = hierarchical_validator.validate(
            user_query, current_proposed_tool_call, AVAILABLE_TOOLS_SCHEMA
        )

        if validation_result.is_valid:
            final_validated_tool_call = validation_result.corrected_call if validation_result.corrected_call else current_proposed_tool_call
            logger.info(f"Validation successful after {retry_count + 1} attempts.")
            logger.info(f"Final validated tool call: {final_validated_tool_call.tool_name}({final_validated_tool_call.args})")
            break # Exit loop, proceed to execution
        else:
            logger.warning(f"Validation failed: {validation_result.error_message}. Attempting correction...")
            
            correction_outcome = correction_mechanisms.apply_correction(
                user_query, current_proposed_tool_call, validation_result, AVAILABLE_TOOLS_SCHEMA
            )

            if correction_outcome.was_corrected:
                current_proposed_tool_call = correction_outcome.corrected_call
                logger.info(f"Correction applied directly. Retrying validation with: {current_proposed_tool_call.tool_name}({current_proposed_tool_call.args})")
                # Loop will continue with the corrected call
            elif correction_outcome.needs_reprompt and correction_outcome.feedback:
                logger.info("Correction mechanism recommended re-prompting Primary Agent.")
                current_proposed_tool_call = primary_agent.re_prompt(
                    user_query, AVAILABLE_TOOLS_SCHEMA, correction_outcome.feedback
                )
                if not current_proposed_tool_call:
                    logger.error("Primary Agent failed to generate a new proposal after re-prompting. Giving up.")
                    break
                logger.info(f"Primary Agent re-prompted proposal: {current_proposed_tool_call.tool_name}({current_proposed_tool_call.args})")
                # Loop will continue with the new proposal
            else:
                logger.error(f"Correction failed after {retry_count + 1} attempts and no further action possible. Giving up on query: '{user_query}'")
                break # Exit loop, no valid call obtained
    else: # This else block executes if the loop completes without a 'break'
        logger.error(f"Failed to achieve a valid tool call after {max_retries} total attempts.")
        return None

    # --- Step 3: Tool Execution ---
    if final_validated_tool_call:
        logger.info("\n--- Executing Final Validated Tool Call ---")
        execution_result = tool_executor.execute_tool_call(final_validated_tool_call)
        logger.info(f"Tool execution result: {execution_result}")
        return execution_result
    else:
        logger.error("No valid tool call obtained, skipping execution.")
        return None

if __name__ == "__main__":
    # --- Example Usage ---

    # Scenario 1: Successful direct proposal and validation
    print("\n\n--- Scenario 1: Valid Query (Weather) ---")
    result1 = run_tool_calling_pipeline("What's the weather like in London?")
    print(f"\nScenario 1 Result: {result1}")

    # Scenario 2: Primary Agent proposes invalid call (missing required param), Validation Agent corrects
    print("\n\n--- Scenario 2: Invalid Query (Meeting - missing date), Validation Agent Corrects ---")
    result2 = run_tool_calling_pipeline("Can you schedule a meeting for Team Sync tomorrow at 10am?")
    print(f"\nScenario 2 Result: {result2}")

    # Scenario 3: Primary Agent proposes invalid call (invalid unit), Hierarchical Validator reports, Correction Mechanisms flags re-prompt, Primary Agent self-corrects
    # Note: For this to work in simulation, ValidationAgent needs to identify the "celsius" in context and SchemaValidator needs an enum check.
    # Let's adjust the tool definition to have 'celsius' as not an enum value, forcing a schema error.
    # Original schema: "enum": ["celsius", "fahrenheit", "metric"]
    # Let's assume the LLM output 'celsius' (valid) but the VALIDATION AGENT determines 'metric' is better based on *its* logic.
    # Or, let's create a situation where the SCHEMA validator finds an invalid unit.
    # We will simulate a case where Primary agent uses an invalid unit like "celcius" (typo) or a unit not in enum.
    # For now, let's simulate a schema validation failure for `unit` if `celsius` is not allowed.
    # Let's make "celsius" NOT in enum for this test, for SchemaValidator to catch.
    # Modifying the global schema for demonstration purposes only.
    # For a realistic setup, schemas are loaded once.
    # Let's make 'weather' tool use `unit_type` instead of `unit` and enum value `metric` but primary agent outputs `unit: "celsius"`

    temp_tool_schema = AVAILABLE_TOOLS_SCHEMA.copy()
    temp_tool_schema["get_current_weather_typo_test"] = {
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["imperial", "metric"], "description": "The unit of temperature to use."},
            },
            "required": ["location"],
        },
    }

    class PrimaryAgentTypoTest(PrimaryAgent):
        def propose_tool_call(self, user_query: str, available_tools_schema: Dict[str, Any]) -> Optional[ToolCall]:
            if "weather in Paris" in user_query.lower():
                return ToolCall(tool_name="get_current_weather_typo_test", args={"location": "Paris", "unit": "celsius"}) # Intentional invalid unit
            return super().propose_tool_call(user_query, available_tools_schema)
        
        def re_prompt(self, user_query: str, available_tools_schema: Dict[str, Any], feedback: str) -> Optional[ToolCall]:
            if "invalid value 'celsius' for parameter 'unit'" in feedback:
                logger.info("PrimaryAgentTypoTest self-corrected 'get_current_weather_typo_test' unit.")
                return ToolCall(tool_name="get_current_weather_typo_test", args={"location": "Paris", "unit": "metric"})
            return super().re_prompt(user_query, available_tools_schema, feedback)

    # Re-run pipeline with adjusted components for scenario 3
    print("\n\n--- Scenario 3: Invalid Unit (celsius) - Primary Agent Re-prompted ---")
    
    # Temporarily override components for this scenario
    original_primary_agent = PrimaryAgent
    original_hierarchical_validator = HierarchicalValidator

    # Patch the global components for this specific test run
    # This is a hack for demonstration, in a real system, you'd inject dependencies
    
    # We can't directly replace `run_tool_calling_pipeline`'s local variables,
    # so we'll just demonstrate the previous successful scenarios.
    # For a real scenario 3, the `PrimaryAgent` would need to be passed in,
    # and the `CorrectionMechanisms` would need more sophisticated logic to provide
    # the exact feedback for re-prompting based on schema validation enum failures.

    # Given the current setup, let's stick to the scenarios easily handled.
    # Scenario 3 will be demonstrated as ValidationAgent correcting a unit,
    # even if schema is technically valid, because it prefers 'metric'.
    # This shows the LLM-based validation agent acting as a higher-level reviewer.
    print("\n\n--- Scenario 3: Validation Agent Prefers 'metric' unit ---")
    result3 = run_tool_calling_pipeline("What's the weather in London in celsius?") # Primary agent might use celsius, Validation agent corrects to metric
    print(f"\nScenario 3 Result: {result3}")


    # Scenario 4: Non-existent tool, Correction Mechanisms flags re-prompt, Primary Agent fails to self-correct (simulated)
    print("\n\n--- Scenario 4: Non-existent Tool, Primary Agent fails to correct ---")
    result4 = run_tool_calling_pipeline("I want to use a non_existent_tool to do something.")
    print(f"\nScenario 4 Result: {result4}")

    # Scenario 5: Valid email sending
    print("\n\n--- Scenario 5: Valid Email Sending ---")
    result5 = run_tool_calling_pipeline("Send an email to john.doe@example.com with subject 'Meeting' and body 'Please confirm your availability.'")
    print(f"\nScenario 5 Result: {result5}")

    # Scenario 6: Primary agent proposes a call, schema validation passes, but Validation agent makes a slight adjustment (e.g., adding a default duration)
    # To demonstrate this, let's have PrimaryAgent not provide duration for schedule_event, and ValidationAgent adds it.
    class PrimaryAgentScenario6(PrimaryAgent):
        def propose_tool_call(self, user_query: str, available_tools_schema: Dict[str, Any]) -> Optional[ToolCall]:
            if "schedule a quick meeting" in user_query.lower():
                # Primary agent omits optional duration_minutes
                return ToolCall(tool_name="schedule_event", args={"title": "Quick Chat", "date": "2023-11-25", "time": "14:00"})
            return super().propose_tool_call(user_query, available_tools_schema)
        
    class ValidationAgentScenario6(ValidationAgent):
        def review_and_correct(self, user_query: str, proposed_tool_call: ToolCall,
                               tool_schemas: Dict[str, Any], schema_validation_feedback: Optional[str]) -> ValidationResult:
            # If schema validation passed but "duration_minutes" is missing for "schedule_event"
            if proposed_tool_call.tool_name == "schedule_event" and "duration_minutes" not in proposed_tool_call.args:
                logger.info("ValidationAgentScenario6 adding default duration_minutes for schedule_event.")
                corrected_call = ToolCall(
                    tool_name=proposed_tool_call.tool_name,
                    args={**proposed_tool_call.args, "duration_minutes": 30}
                )
                return ValidationResult(is_valid=True, corrected_call=corrected_call, error_message="Validation agent added default duration.")
            return super().review_and_correct(user_query, proposed_tool_call, tool_schemas, schema_validation_feedback)

    # Re-initializing parts of the pipeline for Scenario 6
    # This shows how a real system would manage injecting dependencies
    print("\n\n--- Scenario 6: Validation Agent adds default duration ---")
    
    # Temporarily creating and injecting new agents for this specific test
    llm_config_s6 = {"model": "gpt-4-turbo", "temperature": 0.7}
    primary_agent_s6 = PrimaryAgentScenario6(llm_config_s6)
    schema_validator_s6 = SchemaValidator()
    validation_agent_s6 = ValidationAgentScenario6(llm_config_s6)
    hierarchical_validator_s6 = HierarchicalValidator(schema_validator_s6, validation_agent_s6)
    correction_mechanisms_s6 = CorrectionMechanisms()
    tool_executor_s6 = ToolExecutor()

    def run_tool_calling_pipeline_s6(user_query: str, max_retries: int = 3) -> Optional[Any]:
        logger.info(f"\n--- Starting pipeline (Scenario 6) for query: '{user_query}' ---")
        current_proposed_tool_call: Optional[ToolCall] = primary_agent_s6.propose_tool_call(user_query, AVAILABLE_TOOLS_SCHEMA)

        if not current_proposed_tool_call:
            logger.error("Initial proposal from Primary Agent was empty. Cannot proceed.")
            return None
        
        logger.info(f"Primary Agent initial proposal: {current_proposed_tool_call.tool_name}({current_proposed_tool_call.args})")

        for retry_count in range(max_retries):
            logger.info(f"\n--- Validation Attempt {retry_count + 1}/{max_retries} (Scenario 6) ---")
            
            if not current_proposed_tool_call:
                logger.error("No proposed tool call available for validation. Exiting loop.")
                break

            validation_result = hierarchical_validator_s6.validate(
                user_query, current_proposed_tool_call, AVAILABLE_TOOLS_SCHEMA
            )

            if validation_result.is_valid:
                final_validated_tool_call = validation_result.corrected_call if validation_result.corrected_call else current_proposed_tool_call
                logger.info(f"Validation successful after {retry_count + 1} attempts.")
                logger.info(f"Final validated tool call: {final_validated_tool_call.tool_name}({final_validated_tool_call.args})")
                
                # Check if duration_minutes was added by validation agent
                if final_validated_tool_call.tool_name == "schedule_event" and "duration_minutes" in final_validated_tool_call.args:
                    logger.info("SUCCESS: Validation Agent added 'duration_minutes' as expected.")
                break
            else:
                logger.warning(f"Validation failed: {validation_result.error_message}. Attempting correction...")
                
                correction_outcome = correction_mechanisms_s6.apply_correction(
                    user_query, current_proposed_tool_call, validation_result, AVAILABLE_TOOLS_SCHEMA
                )

                if correction_outcome.was_corrected:
                    current_proposed_tool_call = correction_outcome.corrected_call
                    logger.info(f"Correction applied directly. Retrying validation with: {current_proposed_tool_call.tool_name}({current_proposed_tool_call.args})")
                elif correction_outcome.needs_reprompt and correction_outcome.feedback:
                    logger.info("Correction mechanism recommended re-prompting Primary Agent.")
                    current_proposed_tool_call = primary_agent_s6.re_prompt(
                        user_query, AVAILABLE_TOOLS_SCHEMA, correction_outcome.feedback
                    )
                    if not current_proposed_tool_call:
                        logger.error("Primary Agent failed to generate a new proposal after re-prompting. Giving up.")
                        break
                    logger.info(f"Primary Agent re-prompted proposal: {current_proposed_tool_call.tool_name}({current_proposed_tool_call.args})")
                else:
                    logger.error(f"Correction failed after {retry_count + 1} attempts and no further action possible. Giving up on query: '{user_query}'")
                    break
        else:
            logger.error(f"Failed to achieve a valid tool call after {max_retries} total attempts (Scenario 6).")
            return None
        
        if final_validated_tool_call:
            logger.info("\n--- Executing Final Validated Tool Call (Scenario 6) ---")
            execution_result = tool_executor_s6.execute_tool_call(final_validated_tool_call)
            logger.info(f"Tool execution result: {execution_result}")
            return execution_result
        else:
            logger.error("No valid tool call obtained, skipping execution (Scenario 6).")
            return None


    result6 = run_tool_calling_pipeline_s6("schedule a quick meeting for me on 2023-11-25 at 2pm")
    print(f"\nScenario 6 Result: {result6}")
```