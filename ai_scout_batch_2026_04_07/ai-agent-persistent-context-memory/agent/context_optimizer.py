```python
import logging
from typing import Dict, Any, Optional

# Assuming these modules exist as per the architecture.
# We include dummy classes here for isolated testing and to prevent ImportError
# if the other parts of the system are not yet fully implemented.
try:
    from context.context_aggregator import ContextAggregator
    from utils.llm_api import LLM_API_Interface
except ImportError:
    class ContextAggregator:
        """Dummy ContextAggregator for isolated testing."""
        _max_context_tokens: int = 4096
        _summarization_depth: float = 0.5
        _source_weights: Dict[str, float] = {'semantic_retrieval': 0.33, 'changelog_processor': 0.33, 'tree_context_model': 0.34}

        def set_max_context_tokens(self, tokens: int):
            self._max_context_tokens = tokens
            logging.warning(f"Dummy ContextAggregator: set_max_context_tokens called with {tokens}.")

        def set_summarization_depth(self, depth: float):
            self._summarization_depth = depth
            logging.warning(f"Dummy ContextAggregator: set_summarization_depth called with {depth:.2f}.")

        def set_source_weights(self, weights: Dict[str, float]):
            self._source_weights = weights
            logging.warning(f"Dummy ContextAggregator: set_source_weights called with {weights}.")

        def get_current_max_context_tokens(self) -> int:
            return self._max_context_tokens

        def get_current_summarization_depth(self) -> float:
            return self._summarization_depth

        def get_current_source_weights(self) -> Dict[str, float]:
            return self._source_weights

    class LLM_API_Interface:
        """Dummy LLM_API_Interface for isolated testing."""
        def get_model_context_window(self, model_name: str) -> int:
            # Default context window for a common LLM
            return 8192

        def get_token_cost(self, model_name: str, tokens: int) -> float:
            # A placeholder for token cost calculation
            return tokens * 0.000002

        def get_default_model(self) -> str:
            return "gpt-4-turbo-preview"

    logging.warning("Could not import ContextAggregator or LLM_API_Interface. Using dummy classes.")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextOptimizer:
    """
    Dynamically fine-tunes the context provided to the LLM based on observed
    agent performance metrics (e.g., task completion, human feedback, token efficiency).

    Adjusts parameters such as context window size, summarization depth, and the
    weighting of different context sources from the ContextAggregator to deliver
    the most pertinent and concise context, improving LLM efficiency.
    """

    # Default and boundary values for context parameters
    DEFAULT_MAX_TOKENS = 8192  # Initial target max tokens for the context window
    MIN_CONTEXT_TOKENS = 1024  # Absolute minimum context window to maintain basic coherence
    MAX_CONTEXT_TOKENS_GLOBAL = 128000 # Upper bound for context, independent of specific LLM limits

    DEFAULT_SUMMARIZATION_DEPTH = 0.5  # 0.0 (minimal summary, max detail) to 1.0 (max summary, minimal detail)
    MIN_SUMMARIZATION_DEPTH = 0.1      # Always keep some detail
    MAX_SUMMARIZATION_DEPTH = 0.9      # Avoid excessive summarization losing critical info

    # Default weights for context sources (sum should ideally be 1.0)
    DEFAULT_SOURCE_WEIGHTS = {
        'semantic_retrieval': 0.4,
        'changelog_processor': 0.3,
        'tree_context_model': 0.3,
    }

    def __init__(self, context_aggregator: ContextAggregator, llm_api: LLM_API_Interface):
        """
        Initializes the ContextOptimizer.

        Args:
            context_aggregator: An instance of ContextAggregator to dynamically
                                modify its context assembly parameters.
            llm_api: An instance of LLM_API_Interface to query LLM capabilities
                     (e.g., maximum context window for the chosen model).
        """
        if not isinstance(context_aggregator, ContextAggregator):
            raise TypeError("context_aggregator must be an instance of ContextAggregator")
        if not isinstance(llm_api, LLM_API_Interface):
            raise TypeError("llm_api must be an instance of LLM_API_Interface")

        self.context_aggregator = context_aggregator
        self.llm_api = llm_api

        # Get the LLM model name and its specific maximum context window
        self.llm_model_name = self.llm_api.get_default_model()
        self.llm_max_context_window = self.llm_api.get_model_context_window(self.llm_model_name)

        # Initialize current parameters, prioritizing existing aggregator state or defaults
        self._current_max_context_tokens = self.context_aggregator.get_current_max_context_tokens()
        if self._current_max_context_tokens == ContextAggregator._max_context_tokens: # Check if it's the dummy default
            self._current_max_context_tokens = min(self.DEFAULT_MAX_TOKENS, self.llm_max_context_window, self.MAX_CONTEXT_TOKENS_GLOBAL)

        self._current_summarization_depth = self.context_aggregator.get_current_summarization_depth()
        if self._current_summarization_depth == ContextAggregator._summarization_depth: # Check if it's the dummy default
            self._current_summarization_depth = self.DEFAULT_SUMMARIZATION_DEPTH

        self._current_source_weights = self.context_aggregator.get_current_source_weights()
        if self._current_source_weights == ContextAggregator._source_weights: # Check if it's the dummy default
            self._current_source_weights = self.DEFAULT_SOURCE_WEIGHTS.copy()


        logger.info(f"ContextOptimizer initialized. LLM model: {self.llm_model_name}, "
                    f"LLM max context: {self.llm_max_context_window} tokens. "
                    f"Initial context tokens: {self._current_max_context_tokens}, "
                    f"Initial summarization depth: {self._current_summarization_depth:.2f}.")

    def _validate_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Helper method to validate the structure and range of incoming performance metrics.
        Returns True if metrics are valid, False otherwise.
        """
        required_keys = ['task_success_rate', 'human_feedback_score', 'token_efficiency_score', 'recent_context_sources_impact']
        for key in required_keys:
            if key not in metrics:
                logger.error(f"Missing required metric key: '{key}'. Cannot optimize context.")
                return False

        if not (0.0 <= metrics['task_success_rate'] <= 1.0):
            logger.warning(f"task_success_rate {metrics['task_success_rate']:.2f} out of expected range (0.0-1.0).")
            # We can still proceed, but log a warning. For critical errors, return False.
        if not (-1.0 <= metrics['human_feedback_score'] <= 1.0): # -1.0 (negative) to 1.0 (positive)
            logger.warning(f"human_feedback_score {metrics['human_feedback_score']:.2f} out of expected range (-1.0-1.0).")
        if not isinstance(metrics['token_efficiency_score'], (int, float)):
            logger.error(f"token_efficiency_score is not numeric: {metrics['token_efficiency_score']}. Cannot optimize context.")
            return False
        if not isinstance(metrics['recent_context_sources_impact'], dict):
            logger.error(f"recent_context_sources_impact is not a dictionary: {metrics['recent_context_sources_impact']}. Cannot optimize context.")
            return False

        return True

    def optimize_context_parameters(self, performance_metrics: Dict[str, Any]):
        """
        Observes agent performance metrics and dynamically adjusts context parameters
        (context window size, summarization depth, context source weights).

        Args:
            performance_metrics: A dictionary containing various performance indicators:
                - 'task_success_rate' (float): 0.0 to 1.0, higher is better.
                - 'human_feedback_score' (float): -1.0 (negative) to 1.0 (positive), higher is better.
                - 'token_efficiency_score' (float): e.g., average tokens per turn/step, lower is better.
                - 'recent_context_sources_impact' (Dict[str, float]): Impact score for each context source.
                                                                        Higher value means more useful.
        """
        if not self._validate_metrics(performance_metrics):
            return

        logger.info(f"Initiating context optimization with metrics: "
                    f"Success: {performance_metrics['task_success_rate']:.2f}, "
                    f"Feedback: {performance_metrics['human_feedback_score']:.2f}, "
                    f"Token Efficiency: {performance_metrics['token_efficiency_score']:.2f}.")

        # 1. Adjust context window size
        self._adjust_context_window(
            task_success_rate=performance_metrics['task_success_rate'],
            token_efficiency_score=performance_metrics['token_efficiency_score']
        )

        # 2. Adjust summarization depth
        self._adjust_summarization_depth(
            human_feedback_score=performance_metrics['human_feedback_score'],
            task_success_rate=performance_metrics['task_success_rate']
        )

        # 3. Adjust context source weights
        self._adjust_source_weights(
            recent_context_sources_impact=performance_metrics['recent_context_sources_impact']
        )

        logger.info("Context optimization cycle complete.")

    def _adjust_context_window(self, task_success_rate: float, token_efficiency_score: float):
        """
        Adjusts the maximum context window size based on task success and token efficiency.
        Lower token_efficiency_score indicates better efficiency (fewer tokens used).
        """
        current_tokens = self._current_max_context_tokens
        new_tokens = current_tokens
        adjustment_factor = 0.05 # percentage change

        # Heuristic 1: If task success is low, try increasing context to provide more info
        if task_success_rate < 0.6:
            new_tokens = current_tokens * (1 + adjustment_factor)
            logger.debug(f"Task success rate low ({task_success_rate:.2f}). Proposing context increase.")
        # Heuristic 2: If task success is high AND token efficiency is poor (high score), try decreasing context
        elif task_success_rate > 0.8 and token_efficiency_score > 0.5: # Assuming >0.5 is poor efficiency
            new_tokens = current_tokens * (1 - adjustment_factor)
            logger.debug(f"Task success high ({task_success_rate:.2f}) but token efficiency poor ({token_efficiency_score:.2f}). Proposing context decrease.")
        # Heuristic 3: If task success is high AND token efficiency is good (low score), maintain or slightly increase for exploration
        elif task_success_rate > 0.8 and token_efficiency_score < 0.2: # Assuming <0.2 is good efficiency
            new_tokens = current_tokens * (1 + adjustment_factor * 0.5) # Smaller increase
            logger.debug(f"Task success high ({task_success_rate:.2f}) and token efficiency good ({token_efficiency_score:.2f}). Proposing slight context increase for exploration.")
        else:
            logger.debug(f"Context window ({current_tokens}) maintained. Success: {task_success_rate:.2f}, Token Eff: {token_efficiency_score:.2f}.")

        # Apply global and LLM-specific boundaries
        new_tokens = int(max(
            self.MIN_CONTEXT_TOKENS,
            min(new_tokens, self.llm_max_context_window, self.MAX_CONTEXT_TOKENS_GLOBAL)
        ))

        if new_tokens != self._current_max_context_tokens:
            self._current_max_context_tokens = new_tokens
            try:
                self.context_aggregator.set_max_context_tokens(self._current_max_context_tokens)
                logger.info(f"Adjusted max context tokens to: {self._current_max_context_tokens}.")
            except AttributeError as e:
                logger.error(f"ContextAggregator is missing 'set_max_context_tokens' method: {e}")
        else:
            logger.debug("Max context tokens remained unchanged.")

    def _adjust_summarization_depth(self, human_feedback_score: float, task_success_rate: float):
        """
        Adjusts the depth of summarization for context components.
        Higher depth means more summarization (less detail).
        """
        current_depth = self._current_summarization_depth
        new_depth = current_depth
        adjustment_amount = 0.05 # amount to change depth by

        # Heuristic 1: If human feedback is negative or task success is low, decrease summarization (more detail needed)
        if human_feedback_score < 0.0 or task_success_rate < 0.6:
            new_depth = current_depth - adjustment_amount
            logger.debug(f"Negative feedback ({human_feedback_score:.2f}) or low success ({task_success_rate:.2f}). Proposing decreased summarization depth.")
        # Heuristic 2: If human feedback is positive and task success is high, try increasing summarization (conciseness)
        elif human_feedback_score > 0.5 and task_success_rate > 0.8:
            new_depth = current_depth + adjustment_amount
            logger.debug(f"Positive feedback ({human_feedback_score:.2f}) and high success ({task_success_rate:.2f}). Proposing increased summarization depth.")
        else:
            logger.debug(f"Summarization depth ({current_depth:.2f}) maintained. Feedback: {human_feedback_score:.2f}, Success: {task_success_rate:.2f}.")

        # Apply boundaries
        new_depth = max(self.MIN_SUMMARIZATION_DEPTH, min(new_depth, self.MAX_SUMMARIZATION_DEPTH))

        if abs(new_depth - self._current_summarization_depth) > 0.01: # Check for significant change
            self._current_summarization_depth = new_depth
            try:
                self.context_aggregator.set_summarization_depth(self._current_summarization_depth)
                logger.info(f"Adjusted summarization depth to: {self._current_summarization_depth:.2f}.")
            except AttributeError as e:
                logger.error(f"ContextAggregator is missing 'set_summarization_depth' method: {e}")
        else:
            logger.debug("Summarization depth remained largely unchanged.")

    def _adjust_source_weights(self, recent_context_sources_impact: Dict[str, float]):
        """
        Adjusts the weighting of different context sources based on their recent impact.
        Sources with higher impact get increased weight.
        """
        current_weights = self._current_source_weights.copy()
        new_weights = current_weights.copy()
        
        # Calculate total impact from provided metrics
        total_impact = sum(recent_context_sources_impact.values())

        if total_impact == 0:
            logger.warning("No measurable impact for context sources provided. Maintaining current weights.")
            return

        # Blend new impact-based adjustments with current weights to prevent drastic swings
        # A higher 'momentum' factor (e.g., 0.7) means current weights have more influence
        # A lower 'learning_rate' factor (e.g., 0.3) means new impact has less influence on a single step
        momentum_factor = 0.7
        learning_rate = 0.3

        for source, impact in recent_context_sources_impact.items():
            if source in new_weights:
                # Calculate the desired new weight based on its proportion of total impact
                target_weight_from_impact = impact / total_impact
                # Blend with current weight
                new_weights[source] = current_weights[source] * momentum_factor + target_weight_from_impact * learning_rate
            else:
                logger.warning(f"Impact reported for unknown context source: '{source}'. Skipping adjustment for it.")
                # Optionally, add new sources with a small initial weight
                # new_weights[source] = 0.01 # Example

        # Re-normalize weights to ensure they sum to 1.0 (or very close)
        sum_new_weights = sum(new_weights.values())
        if sum_new_weights > 0:
            new_weights = {source: weight / sum_new_weights for source, weight in new_weights.items()}
        else:
            logger.error("All context source weights became zero after adjustment. Resetting to defaults.")
            new_weights = self.DEFAULT_SOURCE_WEIGHTS.copy()

        # Check for significant changes before updating
        weights_changed = False
        for source, weight in new_weights.items():
            if abs(current_weights.get(source, 0.0) - weight) > 0.01: # Threshold for significant change
                weights_changed = True
                break

        if weights_changed:
            self._current_source_weights = new_weights
            try:
                self.context_aggregator.set_source_weights(self._current_source_weights)
                logger.info(f"Adjusted context source weights to: {self._current_source_weights}.")
            except AttributeError as e:
                logger.error(f"ContextAggregator is missing 'set_source_weights' method: {e}")
        else:
            logger.debug("Context source weights remained largely unchanged.")


if __name__ == '__main__':
    # Example Usage for testing the ContextOptimizer in isolation
    print("--- Initializing ContextOptimizer ---")
    dummy_aggregator = ContextAggregator()
    dummy_llm_api = LLM_API_Interface()
    optimizer = ContextOptimizer(context_aggregator=dummy_aggregator, llm_api=dummy_llm_api)

    # --- Scenario 1: High success, but inefficient token usage ---
    print("\n--- Running optimization: High Success, High Token Usage ---")
    metrics_scenario1 = {
        'task_success_rate': 0.9,
        'human_feedback_score': 0.7,
        'token_efficiency_score': 0.8, # Higher score means worse efficiency (more tokens used)
        'recent_context_sources_impact': {
            'semantic_retrieval': 0.6,
            'changelog_processor': 0.2,
            'tree_context_model': 0.9, # Tree model was very impactful
        }
    }
    optimizer.optimize_context_parameters(metrics_scenario1)
    print(f"  Optimized Context Tokens: {optimizer._current_max_context_tokens}")
    print(f"  Optimized Summarization Depth: {optimizer._current_summarization_depth:.2f}")
    print(f"  Optimized Source Weights: {optimizer._current_source_weights}")
    # Expected: Context tokens decrease, summarization might increase, tree_context_model weight increases.

    # --- Scenario 2: Low success, negative human feedback ---
    print("\n--- Running optimization: Low Success, Negative Feedback ---")
    metrics_scenario2 = {
        'task_success_rate': 0.3,
        'human_feedback_score': -0.6,
        'token_efficiency_score': 0.2, # Good efficiency (low tokens), but failing
        'recent_context_sources_impact': {
            'semantic_retrieval': 0.1,
            'changelog_processor': 0.7, # Changelog seems to have been more useful
            'tree_context_model': 0.3,
        }
    }
    optimizer.optimize_context_parameters(metrics_scenario2)
    print(f"  Optimized Context Tokens: {optimizer._current_max_context_tokens}")
    print(f"  Optimized Summarization Depth: {optimizer._current_summarization_depth:.2f}")
    print(f"  Optimized Source Weights: {optimizer._current_source_weights}")
    # Expected: Context tokens increase, summarization decreases (more detail), changelog_processor weight increases.

    # --- Scenario 3: Stable, good performance ---
    print("\n--- Running optimization: Stable, Good Performance ---")
    metrics_scenario3 = {
        'task_success_rate': 0.85,
        'human_feedback_score': 0.6,
        'token_efficiency_score': 0.15, # Very good efficiency
        'recent_context_sources_impact': {
            'semantic_retrieval': 0.4,
            'changelog_processor': 0.3,
            'tree_context_model': 0.3,
        }
    }
    optimizer.optimize_context_parameters(metrics_scenario3)
    print(f"  Optimized Context Tokens: {optimizer._current_max_context_tokens}")
    print(f"  Optimized Summarization Depth: {optimizer._current_summarization_depth:.2f}")
    print(f"  Optimized Source Weights: {optimizer._current_source_weights}")
    # Expected: Context parameters remain largely unchanged or slight tweaks.

    # --- Scenario 4: Invalid metrics ---
    print("\n--- Running optimization: Invalid Metrics (missing key) ---")
    metrics_scenario4_invalid = {
        'task_success_rate': 0.7,
        'human_feedback_score': 0.2,
        'token_efficiency_score': 0.3,
        # 'recent_context_sources_impact' is missing
    }
    optimizer.optimize_context_parameters(metrics_scenario4_invalid)
    print("  (Optimization should have logged an error and skipped adjustments.)")
    print(f"  Current Context Tokens: {optimizer._current_max_context_tokens}")
    print(f"  Current Summarization Depth: {optimizer._current_summarization_depth:.2f}")
    print(f"  Current Source Weights: {optimizer._current_source_weights}")
```