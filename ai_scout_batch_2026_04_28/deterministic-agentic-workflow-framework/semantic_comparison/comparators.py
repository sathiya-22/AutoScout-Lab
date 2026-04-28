import abc
import json
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Assuming these relative imports are correct based on the architecture notes
from utils.llm_connector import LLMConnector
import config

logger = logging.getLogger(__name__)

# --- Helper for Comparison Result ---
class ComparisonResult(NamedTuple):
    """
    Represents the outcome of a semantic comparison between two outputs.
    """
    is_equivalent: bool
    score: float
    reason: str
    details: Optional[Dict[str, Any]] = None

# --- Base Comparator Class ---
class BaseComparator(abc.ABC):
    """
    Abstract base class for all semantic comparison strategies.
    Defines the interface for comparing LLM outputs.
    """
    @abc.abstractmethod
    def compare(self, output1: str, output2: str, context: Optional[str] = None) -> ComparisonResult:
        """
        Compares two LLM outputs for semantic equivalence.

        Args:
            output1: The first LLM output string.
            output2: The second LLM output string.
            context: Optional context for the comparison, useful for LLM-based comparators
                     to understand the task's intent.

        Returns:
            A ComparisonResult indicating equivalence, a numeric score, and a reason.
        """
        pass

# --- 1. Embedding-based Similarity Comparator ---
class EmbeddingComparator(BaseComparator):
    """
    Compares LLM outputs based on the cosine similarity of their vector embeddings.
    Requires an LLMConnector capable of generating text embeddings.
    """
    def __init__(self, llm_connector: LLMConnector):
        """
        Initializes the EmbeddingComparator.

        Args:
            llm_connector: An instance of LLMConnector to get embeddings.
        """
        self._llm_connector = llm_connector
        # Threshold for determining equivalence, loaded from config
        self._threshold = getattr(config, 'SEMANTIC_COMPARISON_EMBEDDING_THRESHOLD', 0.8)
        logger.info(f"Initialized EmbeddingComparator with threshold: {self._threshold}")

    def compare(self, output1: str, output2: str, context: Optional[str] = None) -> ComparisonResult:
        try:
            # Get embeddings for both outputs
            embedding1 = self._llm_connector.get_embedding(output1)
            embedding2 = self._llm_connector.get_embedding(output2)

            if embedding1 is None or embedding2 is None:
                return ComparisonResult(
                    is_equivalent=False,
                    score=0.0,
                    reason="Could not generate embeddings for one or both outputs. "
                           "Ensure LLMConnector's embedding service is available and inputs are valid.",
                    details={"error": "Embedding generation failed"}
                )

            # Convert to numpy arrays and reshape for sklearn's cosine_similarity
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)

            # Calculate cosine similarity
            similarity_score = cosine_similarity(vec1, vec2)[0][0]

            is_equivalent = similarity_score >= self._threshold
            reason = (f"Embedding similarity ({similarity_score:.4f}) "
                      f"{'meets' if is_equivalent else 'does not meet'} "
                      f"the configured threshold ({self._threshold:.4f}).")

            return ComparisonResult(
                is_equivalent=is_equivalent,
                score=float(similarity_score), # Ensure float type
                reason=reason,
                details={"similarity_score": float(similarity_score), "threshold": self._threshold}
            )
        except Exception as e:
            logger.error(f"Error in EmbeddingComparator during comparison: {e}", exc_info=True)
            return ComparisonResult(
                is_equivalent=False,
                score=0.0,
                reason=f"An unexpected error occurred during embedding comparison: {type(e).__name__} - {e}",
                details={"error": str(e)}
            )

# --- 2. Structured Diffing Comparator Helper ---
def _recursive_compare_json(obj1: Any, obj2: Any, ignore_list_order: bool = True) -> bool:
    """
    Recursively compares two JSON-like Python objects (dicts/lists/primitives)
    for deep structural equivalence.

    Args:
        obj1: The first object to compare.
        obj2: The second object to compare.
        ignore_list_order: If True, the order of elements in lists is ignored.

    Returns:
        True if the objects are structurally equivalent, False otherwise.
    """
    if type(obj1) != type(obj2):
        return False

    if isinstance(obj1, dict):
        if set(obj1.keys()) != set(obj2.keys()):
            return False
        for key in obj1:
            if not _recursive_compare_json(obj1[key], obj2[key], ignore_list_order):
                return False
        return True
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            return False
        if ignore_list_order:
            # For lists, compare elements ignoring order.
            # This implementation iterates through obj1 and tries to find a match in obj2.
            # It marks matched items in obj2 to ensure each item is matched only once.
            matched_indices = [False] * len(obj2)
            for item1 in obj1:
                found_match = False
                for i, item2 in enumerate(obj2):
                    if not matched_indices[i] and _recursive_compare_json(item1, item2, ignore_list_order):
                        matched_indices[i] = True
                        found_match = True
                        break
                if not found_match:
                    return False
            return True # All items in obj1 found a match in obj2
        else:
            # If order matters, compare element by element
            for item1, item2 in zip(obj1, obj2):
                if not _recursive_compare_json(item1, item2, ignore_list_order):
                    return False
            return True
    else:
        # Primitive types (str, int, float, bool, None)
        return obj1 == obj2

class StructuredComparator(BaseComparator):
    """
    Compares LLM outputs assuming they conform to a structured format (e.g., JSON).
    It parses the outputs and performs a deep structural comparison.
    """
    def __init__(self):
        """
        Initializes the StructuredComparator.
        """
        self._ignore_list_order = getattr(config, 'SEMANTIC_COMPARISON_STRUCTURED_IGNORE_LIST_ORDER', True)
        logger.info(f"Initialized StructuredComparator with ignore_list_order: {self._ignore_list_order}")

    def compare(self, output1: str, output2: str, context: Optional[str] = None) -> ComparisonResult:
        try:
            obj1 = json.loads(output1)
            obj2 = json.loads(output2)
        except json.JSONDecodeError as e:
            # Log specific output causing the error for better debugging
            if "output1" in str(e): # Heuristic to guess which output failed parsing
                error_details = {"error": str(e), "output1_parsable": False, "output2_parsable": True}
            elif "output2" in str(e):
                error_details = {"error": str(e), "output1_parsable": True, "output2_parsable": False}
            else:
                error_details = {"error": str(e), "output1_parsable": False, "output2_parsable": False}

            logger.warning(f"StructuredComparator: One or both outputs failed JSON parsing: {e}")
            return ComparisonResult(
                is_equivalent=False,
                score=0.0,
                reason=f"One or both outputs are not valid JSON: {e}",
                details=error_details
            )
        except Exception as e:
            logger.error(f"StructuredComparator: An unexpected error occurred before comparison: {e}", exc_info=True)
            return ComparisonResult(
                is_equivalent=False,
                score=0.0,
                reason=f"An unexpected error occurred during JSON parsing preparation: {type(e).__name__} - {e}",
                details={"error": str(e)}
            )

        is_equivalent = _recursive_compare_json(obj1, obj2, ignore_list_order=self._ignore_list_order)
        score = 1.0 if is_equivalent else 0.0
        reason = (f"Structured comparison resulted in {'equivalence' if is_equivalent else 'divergence'} "
                  f"(configured to ignore list order: {self._ignore_list_order}).")

        return ComparisonResult(
            is_equivalent=is_equivalent,
            score=score,
            reason=reason,
            details={"ignore_list_order": self._ignore_list_order}
        )

# --- 3. LLM-as-Comparator ---
class LLMAsComparator(BaseComparator):
    """
    Uses an LLM to evaluate the semantic equivalence of two outputs given a task context.
    Leverages the LLM's understanding to determine equivalence.
    """
    def __init__(self, llm_connector: LLMConnector):
        """
        Initializes the LLMAsComparator.

        Args:
            llm_connector: An instance of LLMConnector to interact with an LLM.
        """
        self._llm_connector = llm_connector
        # Model name for the LLM-as-Comparator, loaded from config
        self._model_name = getattr(config, 'SEMANTIC_COMPARISON_LLM_COMPARATOR_MODEL', "gpt-4-turbo")
        logger.info(f"Initialized LLMAsComparator using model: {self._model_name}")

    def compare(self, output1: str, output2: str, context: Optional[str] = None) -> ComparisonResult:
        if not context:
            logger.warning("LLMAsComparator is called without context. Providing a default generic context. "
                           "Consider providing relevant task context for better accuracy.")
            context = "general task or information extraction" # Default context if none provided

        prompt = (
            f"Given the following task context: '{context}', "
            f"carefully evaluate if the two provided outputs are semantically equivalent. "
            f"Semantic equivalence means they convey the same core meaning, intent, or achieve "
            f"the same goal, even if their exact wording differs. "
            f"Respond with 'EQUIVALENT' if they are semantically equivalent, "
            f"and 'DIVERGENT' if they differ significantly in meaning or outcome. "
            f"Follow your decision (EQUIVALENT or DIVERGENT) with a brief, concise explanation (max 75 words) "
            f"justifying your choice. Do not include any other text.\n\n"
            f"Output 1:\n```\n{output1}\n```\n\n"
            f"Output 2:\n```\n{output2}\n```\n\n"
            f"Decision (EQUIVALENT/DIVERGENT):"
        )

        try:
            # Call the LLM with the comparison prompt
            llm_response_obj = self._llm_connector.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self._model_name,
                temperature=0.0, # Aim for deterministic output from the LLM
                max_tokens=250 # Limit response length
            )

            # Ensure llm_response_obj is a string
            if not isinstance(llm_response_obj, str):
                logger.error(f"LLMAsComparator received non-string response: {llm_response_obj}")
                return ComparisonResult(
                    is_equivalent=False,
                    score=0.0,
                    reason="LLM response was not a string.",
                    details={"llm_raw_response": str(llm_response_obj), "model_used": self._model_name}
                )

            response_text = llm_response_obj.strip().upper()

            # Parse the LLM's decision and reason
            is_equivalent = response_text.startswith("EQUIVALENT")
            score = 1.0 if is_equivalent else 0.0

            reason_parts = response_text.split('\n', 1)
            decision_line = reason_parts[0].strip()
            extracted_reason = decision_line
            if len(reason_parts) > 1:
                extracted_reason = reason_parts[1].strip()
            elif len(decision_line.split(' ', 1)) > 1:
                extracted_reason = decision_line.split(' ', 1)[1].strip()

            if not extracted_reason or extracted_reason.startswith(("EQUIVALENT", "DIVERGENT")):
                extracted_reason = "LLM provided no specific justification beyond the decision."

            # Truncate reason if it's excessively long
            if len(extracted_reason) > 200:
                extracted_reason = extracted_reason[:197] + "..."

            return ComparisonResult(
                is_equivalent=is_equivalent,
                score=score,
                reason=f"LLM determined outputs are {('equivalent' if is_equivalent else 'divergent')}. Justification: {extracted_reason}",
                details={"llm_raw_response": llm_response_obj, "model_used": self._model_name}
            )
        except Exception as e:
            logger.error(f"Error in LLMAsComparator during comparison: {e}", exc_info=True)
            return ComparisonResult(
                is_equivalent=False,
                score=0.0,
                reason=f"An unexpected error occurred during LLM-based comparison: {type(e).__name__} - {e}",
                details={"error": str(e), "model_used": self._model_name}
            )