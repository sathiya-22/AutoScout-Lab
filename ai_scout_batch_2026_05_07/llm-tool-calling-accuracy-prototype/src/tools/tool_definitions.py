from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Literal, Dict, Type, Any, Optional
import json

class ToolSchema(BaseModel):
    """
    Base class for all tool schemas, providing common attributes.
    Child classes must override 'name' and 'description'.
    """
    class Config:
        extra = 'forbid' # Ensure no extra fields are passed

    # These will be set as class attributes in concrete tool definitions
    name: str = Field(..., description="The name of the tool, unique identifier.")
    description: str = Field(..., description="A clear and concise description of what the tool does.")

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return cls.description

    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """Returns the JSON schema for the tool, suitable for LLM consumption."""
        schema = cls.schema()
        # Remove the 'name' and 'description' fields from properties, as they are metadata
        # not arguments to the tool function itself.
        if 'properties' in schema:
            schema['properties'] = {
                k: v for k, v in schema['properties'].items() if k not in ['name', 'description']
            }
            if 'required' in schema:
                schema['required'] = [
                    req for req in schema['required'] if req not in ['name', 'description']
                ]
        return schema

# --- Specific Tool Definitions ---

class SearchToolSchema(ToolSchema):
    name: str = "search_web"
    description: "Searches the web for information using a given query."

    query: str = Field(..., description="The search query to use.")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of search results to return.")
    time_period: Optional[Literal["past_hour", "past_day", "past_week", "past_month", "past_year"]] = Field(
        None, description="Filter results by a specific time period."
    )

class SendEmailToolSchema(ToolSchema):
    name: str = "send_email"
    description: "Sends an email to one or more recipients."

    recipients: List[EmailStr] = Field(..., min_items=1, description="A list of email addresses for the primary recipients.")
    subject: str = Field(..., description="The subject line of the email.")
    body: str = Field(..., description="The main content of the email.")
    cc: Optional[List[EmailStr]] = Field(None, description="Optional list of email addresses for CC recipients.")
    attachments: Optional[List[str]] = Field(None, description="Optional list of file paths to attach to the email.")

class CreateCalendarEventToolSchema(ToolSchema):
    name: str = "create_calendar_event"
    description: "Creates a new event in the user's calendar."

    title: str = Field(..., description="The title or subject of the event.")
    start_time: str = Field(..., description="The start time of the event in ISO 8601 format (e.g., '2023-10-27T10:00:00Z').")
    end_time: str = Field(..., description="The end time of the event in ISO 8601 format (e.g., '2023-10-27T11:00:00Z').")
    attendees: Optional[List[EmailStr]] = Field(None, description="Optional list of email addresses of attendees.")
    location: Optional[str] = Field(None, description="Optional physical location for the event.")
    description: Optional[str] = Field(None, description="Optional detailed description for the event.")

    @validator('start_time', 'end_time', pre=True)
    def validate_time_format(cls, v):
        # Basic validation for ISO 8601 format, more robust validation would use datetime parsing
        if not isinstance(v, str) or not (
            len(v) >= 19 and v[4] == '-' and v[7] == '-' and v[10] == 'T' and v[13] == ':' and v[16] == ':'
        ):
            raise ValueError("Time must be in ISO 8601 format (e.g., 'YYYY-MM-DDTHH:MM:SSZ')")
        return v

class GetWeatherToolSchema(ToolSchema):
    name: str = "get_current_weather"
    description: "Retrieves current weather conditions for a specified city."

    location: str = Field(..., description="The city and optionally state/country for which to get weather.")
    unit: Literal["celsius", "fahrenheit"] = Field("celsius", description="The unit of temperature to return.")

class ConvertCurrencyToolSchema(ToolSchema):
    name: str = "convert_currency"
    description: "Converts an amount from one currency to another."

    amount: float = Field(..., gt=0, description="The amount of money to convert.")
    from_currency: str = Field(..., min_length=3, max_length=3, description="The 3-letter currency code to convert from (e.g., 'USD', 'EUR').")
    to_currency: str = Field(..., min_length=3, max_length=3, description="The 3-letter currency code to convert to (e.g., 'GBP', 'JPY').")

# --- Registry of all tools ---

TOOL_DEFINITIONS: Dict[str, Type[ToolSchema]] = {
    SearchToolSchema.get_name(): SearchToolSchema,
    SendEmailToolSchema.get_name(): SendEmailToolSchema,
    CreateCalendarEventToolSchema.get_name(): CreateCalendarEventToolSchema,
    GetWeatherToolSchema.get_name(): GetWeatherToolSchema,
    ConvertCurrencyToolSchema.get_name(): ConvertCurrencyToolSchema,
}

def get_tool_schema(tool_name: str) -> Optional[Type[ToolSchema]]:
    """
    Retrieves the Pydantic schema class for a given tool name.

    Args:
        tool_name: The name of the tool (e.g., "search_web").

    Returns:
        The Pydantic BaseModel class for the tool, or None if not found.
    """
    return TOOL_DEFINITIONS.get(tool_name)

def get_all_tool_json_schemas() -> List[Dict[str, Any]]:
    """
    Returns a list of JSON schemas for all defined tools, suitable for LLM consumption.
    Each item in the list will be formatted as:
    {
        "name": "tool_name",
        "description": "Tool description.",
        "parameters": { ... Pydantic JSON schema for parameters ... }
    }
    """
    json_schemas = []
    for tool_name, tool_class in TOOL_DEFINITIONS.items():
        # The schema generated by .get_json_schema() already has `name` and `description`
        # filtered out of `properties` and `required` arguments.
        # We need to format it for LLM which typically expects `name`, `description` at the top-level
        # and then `parameters` containing the actual Pydantic schema for the args.
        tool_json_schema = tool_class.get_json_schema()
        
        formatted_schema = {
            "name": tool_class.get_name(),
            "description": tool_class.get_description(),
            "parameters": tool_json_schema # This is the actual Pydantic schema for the arguments
        }
        json_schemas.append(formatted_schema)
    return json_schemas

if __name__ == "__main__":
    print("--- Tool Definitions Loaded ---")
    print(f"Total tools defined: {len(TOOL_DEFINITIONS)}\n")

    # Example: Print schema for a specific tool
    search_tool_schema_class = get_tool_schema("search_web")
    if search_tool_schema_class:
        print(f"Schema for '{search_tool_schema_class.get_name()}':")
        print(f"  Description: {search_tool_schema_class.get_description()}")
        print("  Pydantic JSON Schema for parameters:")
        print(json.dumps(search_tool_schema_class.get_json_schema(), indent=2))
        print("\n--- Example Call Validation (SearchTool) ---")
        try:
            # Valid call
            valid_call = search_tool_schema_class(query="latest AI research", max_results=10)
            print(f"Valid call: {valid_call.model_dump_json(indent=2)}")
            # Invalid call (max_results too high)
            invalid_call = search_tool_schema_class(query="test", max_results=25)
        except Exception as e:
            print(f"Invalid call failed as expected: {e}")
            pass # We expect this to fail for demonstration

    print("\n--- All Tool JSON Schemas for LLM ---")
    all_llm_schemas = get_all_tool_json_schemas()
    for schema in all_llm_schemas:
        print(json.dumps(schema, indent=2))
        print("-" * 30)

    # Example: Email tool validation
    email_tool_schema_class = get_tool_schema("send_email")
    if email_tool_schema_class:
        print("\n--- Example Call Validation (SendEmailTool) ---")
        try:
            valid_email = email_tool_schema_class(
                recipients=["test@example.com", "another@domain.org"],
                subject="Hello from AI",
                body="This is a test email.",
                cc=["copy@example.com"]
            )
            print(f"Valid email call: {valid_email.model_dump_json(indent=2)}")

            invalid_email = email_tool_schema_class(
                recipients=["invalid-email"], # Should fail
                subject="Bad Email",
                body="This should not pass."
            )
        except Exception as e:
            print(f"Invalid email call failed as expected: {e}")
            pass
            
    # Example: Calendar event tool validation
    calendar_tool_schema_class = get_tool_schema("create_calendar_event")
    if calendar_tool_schema_class:
        print("\n--- Example Call Validation (CreateCalendarEventTool) ---")
        try:
            valid_event = calendar_tool_schema_class(
                title="Project Sync",
                start_time="2023-11-01T14:00:00Z",
                end_time="2023-11-01T15:00:00Z",
                attendees=["manager@example.com"],
                location="Conference Room A"
            )
            print(f"Valid event call: {valid_event.model_dump_json(indent=2)}")

            invalid_event_time = calendar_tool_schema_class(
                title="Bad Event",
                start_time="2023/11/01 14:00", # Invalid format
                end_time="2023-11-01T15:00:00Z"
            )
        except Exception as e:
            print(f"Invalid event time call failed as expected: {e}")
            pass