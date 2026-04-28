from pydantic import BaseModel, Field, confloat
from enum import Enum

class ComparisonOutcome(str, Enum):
    """
    Represents the semantic outcome of a comparison.
    """
    EQUIVALENT = "equivalent"
    DIVERGENT = "divergent"
    NEEDS_REVIEW = "needs_review"
    ERROR = "error" # Indicates an error occurred during comparison

class SemanticMetric(BaseModel):
    """
    Abstract base class for all semantic evaluation metrics.
    Defines common attributes like name, description, and an ideal target score.
    """
    name: str = Field(..., description="A unique identifier for the metric.")
    description: str = Field("", description="A brief description of what the metric measures.")
    # The ideal target value for a perfect match (e.g., 1.0 for similarity, 0.0 for distance).
    target_value: float | None = Field(None, description="The ideal value indicating a perfect match.")

class CosineSimilarityMetric(SemanticMetric):
    """
    Measures semantic similarity using the cosine distance between vector embeddings.
    A higher score indicates greater similarity.
    """
    name: str = "cosine_similarity"
    description: str = "Measures the cosine similarity between vector embeddings of two outputs, ranging from -1 to 1."
    threshold: confloat(ge=-1.0, le=1.0) = Field(0.8, description="Minimum cosine similarity score to consider outputs semantically equivalent. Default is 0.8.")
    target_value: float = 1.0 # Perfect similarity

class StructuralConsistencyMetric(SemanticMetric):
    """
    Measures the consistency between two structured outputs (e.g., JSON, YAML).
    It can account for schema adherence, missing/extra fields, and value differences.
    A score of 1.0 typically means identical structure and values (with allowed exceptions).
    """
    name: str = "structural_consistency"
    description: str = "Measures how structurally similar two parsed outputs are. Can ignore specified keys or order of lists."
    threshold: confloat(ge=0.0, le=1.0) = Field(0.95, description="Minimum structural consistency score to consider outputs equivalent. Default is 0.95.")
    ignore_keys: list[str] = Field([], description="List of JSON/dict keys (paths) to ignore during structural comparison.")
    ordered_list_comparison: bool = Field(False, description="If True, the order of elements in lists matters for comparison; otherwise, order is ignored.")
    target_value: float = 1.0 # Perfect structural match

class LLMAgreementMetric(SemanticMetric):
    """
    Measures semantic agreement based on an LLM's assessment.
    The LLM is prompted to evaluate equivalence and typically returns a confidence score
    or a categorical judgment.
    """
    name: str = "llm_agreement"
    description: str = "Evaluates semantic equivalence by querying an LLM, potentially returning a confidence score (0-1)."
    agreement_threshold: confloat(ge=0.0, le=1.0) = Field(0.75, description="Minimum confidence/agreement score from the LLM to consider outputs semantically equivalent. Default is 0.75.")
    llm_model: str = Field("gpt-4o", description="The specific LLM model used for the comparison task.")
    target_value: float = 1.0 # Perfect LLM agreement/confidence

class EvaluationResult(BaseModel):
    """
    Encapsulates the complete result of a single semantic comparison using a specific metric.
    """
    metric: SemanticMetric = Field(..., description="The metric used for this evaluation.")
    score: float = Field(..., description="The calculated score obtained from the comparison.")
    is_equivalent: bool = Field(False, description="True if the score meets the metric's equivalence threshold, False otherwise.")
    outcome: ComparisonOutcome = Field(ComparisonOutcome.NEEDS_REVIEW, description="Categorical outcome of the comparison based on the score and thresholds.")
    details: dict = Field({}, description="Additional details or raw outputs from the comparison process (e.g., specific diffs, LLM prompt/response).")
    error: str | None = Field(None, description="Any error message if the comparison failed.")

    def __init__(self, **data):
        super().__init__(**data)
        # Automatically determine `is_equivalent` and `outcome` if not explicitly provided
        if self.error:
            self.outcome = ComparisonOutcome.ERROR
            self.is_equivalent = False
        elif not self.is_equivalent and "score" in data and self.metric.threshold is not None:
            # For most similarity metrics, score >= threshold means equivalent
            # This logic might need to be inverted for 'distance' metrics
            self.is_equivalent = self.score >= self.metric.threshold

        if self.is_equivalent:
            self.outcome = ComparisonOutcome.EQUIVALENT
        elif self.outcome == ComparisonOutcome.NEEDS_REVIEW and not self.error:
            # If not explicitly equivalent and not an error, it's divergent by default
            # unless more nuanced logic for NEEDS_REVIEW is implemented.
            self.outcome = ComparisonOutcome.DIVERGENT