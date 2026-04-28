```python
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, UUID4
import uuid

# Base schema for all entities, providing common fields for traceability and uniqueness.
class BaseSchema(BaseModel):
    """
    Base schema for all Pydantic models in the framework, providing
    common fields like a unique identifier and timestamp.
    """
    id: UUID4 = Field(default_factory=uuid.uuid4, description="Unique identifier for the schema instance.")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of when the schema instance was created."
    )
    # Optional field to link related operations or workflow runs across different schemas
    correlation_id: Optional[UUID4] = Field(
        None,
        description="Optional ID to link related operations or a sequence of actions within a workflow."
    )

# Schema for inputs provided to an agent.
class AgentInput(BaseSchema):
    """
    Defines the structured input for an agent to perform a specific task.
    """
    task_id: UUID4 = Field(..., description="Unique identifier for the specific task this input belongs to.")
    agent_name: str = Field(..., description="The name of the agent to which this input is directed.")
    parameters: Dict[str, Any] = Field(
        ...,
        description="A dictionary of input parameters that the agent will use for its task."
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual information relevant to the task, such as system state or previous steps."
    )

# Schema for outputs produced by an agent.
class AgentOutput(BaseSchema):
    """
    Defines the structured output produced by an agent after completing a task.
    """
    task_id: UUID4 = Field(..., description="Unique identifier for the task this output pertains to.")
    agent_name: str = Field(..., description="The name of the agent that produced this output.")
    output_data: Any = Field(
        ...,
        description="The actual data produced by the agent. Can be any serializable type (dict, list, string, etc.)."
    )
    status: str = Field(
        ...,
        description="The execution status of the agent (e.g., 'success', 'failure', 'partial_success', 'pending')."
    )
    error_message: Optional[str] = Field(
        None,
        description="Optional detailed error message if the status indicates a failure or partial success."
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional performance or outcome metrics related to the agent's execution (e.g., token usage, latency)."
    )

# Schema for an agent's action, encompassing its input, output, and proposed state changes.
class AgentAction(BaseSchema):
    """
    Represents a complete action performed by an agent, including its input,
    resulting output, and any proposed changes to the centralized system state.
    """
    action_id: UUID4 = Field(default_factory=uuid.uuid4, description="Unique identifier for this specific action.")
    agent_name: str = Field(..., description="The name of the agent performing the action.")
    action_type: str = Field(
        ...,
        description="A descriptive string for the type of action (e.g., 'data_ingestion', 'report_generation', 'database_update')."
    )
    input_payload: AgentInput = Field(..., description="The AgentInput that initiated this action.")
    output_payload: Optional[AgentOutput] = Field(
        None,
        description="The AgentOutput resulting from this action. Can be None if the action is pending or failed early."
    )
    # Proposed changes to the system's canonical state, structured by state component.
    state_changes_proposed: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "A dictionary where keys represent state components (e.g., 'documents', 'users', 'tasks'). "
            "Values are dictionaries describing proposed modifications to that component, "
            "such as {'add': [...], 'update': {...}, 'delete_ids': [...]}, or a direct new state fragment."
        )
    )

# Schema for defining the expected semantic outcome of an agent action.
class ExpectedOutcome(BaseSchema):
    """
    Defines the desired semantic state or output an AgentAction should achieve.
    This is crucial for semantic comparison and reconciliation.
    """
    outcome_id: UUID4 = Field(default_factory=uuid.uuid4, description="Unique ID for this expected outcome definition.")
    action_id: Optional[UUID4] = Field(
        None,
        description="Optional ID of the specific AgentAction this outcome relates to. Can be None if it's a general expectation."
    )
    description: str = Field(
        ...,
        description="A natural language description of the expected semantic outcome, aiding human and LLM interpretation."
    )
    expected_output_schema: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Optional JSON schema or Pydantic model dictionary describing the expected structure and "
            "data types of the AgentOutput's 'output_data'. Used for structural validation."
        )
    )
    semantic_criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "A dictionary of criteria for semantic comparison. Examples: "
            "{'keywords_present': ['summary', 'report'], 'entities_present': ['customer_id', 'product_name'], "
            "'min_length': 100, 'sentiment': 'positive_or_neutral'}. These guide the semantic comparators."
        )
    )
    invariants_to_maintain: List[str] = Field(
        default_factory=list,
        description="A list of unique identifiers or descriptions of system-wide invariants or rules that must hold true after the action is performed."
    )
    match_level: str = Field(
        "strict",
        description=(
            "The strictness level for matching this outcome against actual outputs or states. "
            "Examples: 'strict' (requires near-exact semantic match), 'semantic' (allows paraphrasing/minor variations), "
            "'lenient' (only core intent must be met)."
        )
    )

# --- Example Domain-Specific Schemas (Demonstrates Extensibility) ---

class DocumentProcessingInput(AgentInput):
    """
    Specific input schema for an agent designed to process documents.
    """
    document_id: str = Field(..., description="The ID of the document to be processed.")
    content: str = Field(..., description="The textual content of the document.")
    instruction: str = Field(..., description="Specific instructions for processing this document.")

class DocumentSummaryOutput(AgentOutput):
    """
    Specific output schema for an agent that generates document summaries.
    """
    summary: str = Field(..., description="The generated summary of the document.")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords from the document.")
    sentiment: Optional[str] = Field(
        None,
        description="Detected sentiment of the document (e.g., 'positive', 'negative', 'neutral')."
    )

class ProcessDocumentAction(AgentAction):
    """
    A concrete action representing a document processing operation by an agent.
    Overrides generic types with domain-specific ones.
    """
    action_type: str = Field("process_document", const=True) # Enforce action type
    input_payload: DocumentProcessingInput = Field(..., description="Input for document processing.")
    output_payload: Optional[DocumentSummaryOutput] = Field(
        None,
        description="Output from document processing (e.g., summary, keywords)."
    )
    # Example of a specific proposed state change related to documents
    state_changes_proposed: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {"documents": {"updated_status": True}},
        description="Indicates an update to the document's processing status in the state."
    )

class ExpectedDocumentSummaryOutcome(ExpectedOutcome):
    """
    Defines the expected outcome for a document summary task.
    """
    action_id: UUID4 = Field(..., description="The action ID this specific outcome is linked to.")
    description: str = "A concise summary of the document, extracting key entities and identifying overall sentiment accurately."
    expected_output_schema: Dict[str, Any] = Field(
        default_factory=DocumentSummaryOutput.model_json_schema,
        description="JSON schema for the expected DocumentSummaryOutput."
    )
    semantic_criteria: Dict[str, Any] = {
        "summary_length_min": 50,
        "summary_length_max": 200,
        "keywords_match_ratio": 0.7, # e.g., 70% of pre-defined target keywords should be present
        "sentiment_detection_accuracy_threshold": 0.8,
        "contains_key_entities": ["company_name", "product_name"]
    }
    invariants_to_maintain: List[str] = [
        "summary_is_accurate",
        "no_pii_leakage_in_summary",
        "sentiment_is_consistent_with_content"
    ]
    match_level: str = "semantic" # Allow for some flexibility in summary wording, focus on meaning.

```