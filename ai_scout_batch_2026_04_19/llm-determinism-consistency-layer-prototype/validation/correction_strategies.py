```python
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

from ..core.llm_interface import LLMInterface, LLMResponse

logger = logging.getLogger(__name__)

class CorrectionStrategy(ABC):
    """
    Abstract base class for LLM output correction strategies.
    Each strategy attempts to correct an invalid LLM output.
    """
    def __init__(self, llm_interface: LLMInterface, max_retries: int = 1):
        """
        Initializes the CorrectionStrategy.

        Args:
            llm_interface: An instance of LLMInterface to make LLM calls.
            max_retries: The maximum number of attempts for the strategy to correct the output.
                         (Not applicable for strategies like HumanInTheLoop that don't make LLM calls).
        """
        self.llm_interface = llm_interface
        self.max_retries = max_retries

    @abstractmethod
    def correct(
        self,
        original_prompt: str,
        invalid_output: str,
        validation_error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Attempts to correct an invalid LLM output.

        Args:
            original_prompt: The prompt that generated the invalid output.
            invalid_output: The LLM output that failed validation.
            validation_error: An optional description of why the output was invalid.
            context: Additional context that might be useful for correction (e.g., original LLM parameters).
            **kwargs: Additional parameters specific to the strategy.

        Returns:
            A corrected string output if successful, otherwise None.
        """
        pass


class SimpleRepromptStrategy(CorrectionStrategy):
    """
    A simple strategy that re-prompts the LLM with the original prompt.
    This assumes that the initial failure was transient or due to a minor LLM 'slip'.
    """
    def correct(
        self,
        original_prompt: str,
        invalid_output: str,
        validation_error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Re-prompts the LLM with the original prompt for correction.
        """
        logger.info(f"Applying SimpleRepromptStrategy. Retrying original prompt (max {self.max_retries} times).")
        # Extract LLM specific parameters from context for consistency
        llm_params = context.get('llm_params', {}) if context else {}

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retries} with original prompt.")
                response: LLMResponse = self.llm_interface.generate(prompt=original_prompt, **llm_params)
                return response.text
            except Exception as e:
                logger.warning(f"Error during SimpleRepromptStrategy attempt {attempt + 1}: {e}", exc_info=True)
        logger.warning(f"SimpleRepromptStrategy failed after {self.max_retries} attempts for prompt: {original_prompt[:100]}...")
        return None


class SelfCorrectionStrategy(CorrectionStrategy):
    """
    A strategy that prompts the LLM to self-correct by providing the invalid output
    and the reason for its invalidity. This leverages the LLM's ability to reason
    about and fix its own mistakes.
    """
    def __init__(self, llm_interface: LLMInterface, max_retries: int = 1, correction_prompt_template: Optional[str] = None):
        super().__init__(llm_interface, max_retries)
        self.correction_prompt_template = correction_prompt_template or (
            "You previously generated an output for the following prompt that was invalid:\n\n"
            "Original Prompt:\n```\n{original_prompt}\n```\n\n"
            "Your previous (invalid) output:\n```\n{invalid_output}\n```\n\n"
            "The output was invalid because: {validation_error}\n\n"
            "Please carefully review the original prompt, your invalid output, and the reason for its invalidity.\n"
            "Then, generate a *corrected* output that strictly adheres to the requirements. Ensure the corrected output is valid.\n"
            "Corrected output:"
        )

    def correct(
        self,
        original_prompt: str,
        invalid_output: str,
        validation_error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Prompts the LLM to self-correct based on the validation error.
        """
        logger.info(f"Applying SelfCorrectionStrategy. Retrying with self-correction prompt (max {self.max_retries} times).")
        error_msg = validation_error if validation_error else "It did not meet the expected format or content requirements."
        llm_params = context.get('llm_params', {}) if context else {}

        for attempt in range(self.max_retries):
            correction_prompt = self.correction_prompt_template.format(
                original_prompt=original_prompt,
                invalid_output=invalid_output,
                validation_error=error_msg
            )
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retries} with self-correction prompt.")
                response: LLMResponse = self.llm_interface.generate(prompt=correction_prompt, **llm_params)
                return response.text
            except Exception as e:
                logger.warning(f"Error during SelfCorrectionStrategy attempt {attempt + 1}: {e}", exc_info=True)
        logger.warning(f"SelfCorrectionStrategy failed after {self.max_retries} attempts for prompt: {original_prompt[:100]}...")
        return None


class HumanInTheLoopStrategy(CorrectionStrategy):
    """
    A strategy that flags the output for human review and correction.
    This strategy does not return a corrected output directly but indicates
    that human intervention is required, typically by triggering a notification
    or task creation via a callback.
    """
    def __init__(self, llm_interface: LLMInterface, notification_callback: Optional[callable] = None):
        # Human-in-the-loop doesn't involve automatic LLM retries for correction,
        # so max_retries for LLM calls is not applicable here.
        super().__init__(llm_interface, max_retries=0)
        self.notification_callback = notification_callback

    def correct(
        self,
        original_prompt: str,
        invalid_output: str,
        validation_error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Flags the invalid output for human review and handles notification.
        """
        logger.warning(
            f"Applying HumanInTheLoopStrategy. Invalid output requires human review."
            f"\nOriginal Prompt (truncated): {original_prompt[:200]}..."
            f"\nInvalid Output (truncated): {invalid_output[:200]}..."
            f"\nValidation Error: {validation_error}"
        )
        if self.notification_callback:
            try:
                # The callback should handle how to notify or store the task for human review
                self.notification_callback(
                    original_prompt=original_prompt,
                    invalid_output=invalid_output,
                    validation_error=validation_error,
                    context=context
                )
                logger.info("Human-in-the-loop notification triggered successfully.")
            except Exception as e:
                logger.error(f"Error executing human-in-the-loop notification callback: {e}", exc_info=True)
        else:
            logger.warning("No notification_callback provided for HumanInTheLoopStrategy. "
                           "Human intervention indicated but not acted upon programmatically. "
                           "Consider implementing a callback for production systems.")

        # This strategy does not produce a corrected output automatically.
        # It signifies that external action is needed. The caller should interpret None here
        # as "correction pending" or "manual intervention required".
        return None


class SequentialCorrectionStrategy(CorrectionStrategy):
    """
    A composite strategy that tries a sequence of other correction strategies
    until one succeeds or all fail. This allows for a robust, multi-stage correction pipeline.
    """
    def __init__(self, llm_interface: LLMInterface, strategies_config: List[Dict[str, Any]]):
        # The Sequential strategy itself doesn't make direct LLM calls or have retries,
        # so `max_retries` is not directly relevant here for the composite itself.
        # It delegates to its sub-strategies.
        super().__init__(llm_interface, max_retries=1) # Default max_retries for parent, not directly used

        self.sub_strategies: List[CorrectionStrategy] = []
        if not strategies_config:
            raise ValueError("SequentialCorrectionStrategy requires at least one sub-strategy configuration.")

        for i, config in enumerate(strategies_config):
            try:
                if not isinstance(config, dict) or "name" not in config:
                    raise ValueError(f"Sub-strategy config at index {i} is invalid: {config}")
                sub_strategy = _create_strategy_instance(config["name"], llm_interface, **{k: v for k, v in config.items() if k != "name"})
                self.sub_strategies.append(sub_strategy)
            except Exception as e:
                logger.error(f"Failed to initialize sub-strategy from config: {config}. Error: {e}", exc_info=True)
                raise # Re-raise to indicate a configuration issue during init

    def correct(
        self,
        original_prompt: str,
        invalid_output: str,
        validation_error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Attempts correction by iterating through configured sub-strategies.
        """
        logger.info(f"Applying SequentialCorrectionStrategy with {len(self.sub_strategies)} sub-strategies.")
        for i, strategy in enumerate(self.sub_strategies):
            logger.info(f"Attempting correction with sub-strategy {i+1}: {strategy.__class__.__name__}")
            try:
                # Pass all context and kwargs to sub-strategies
                corrected_output = strategy.correct(
                    original_prompt=original_prompt,
                    invalid_output=invalid_output,
                    validation_error=validation_error,
                    context=context,
                    **kwargs
                )
                if corrected_output is not None:
                    logger.info(f"Correction successful using {strategy.__class__.__name__}.")
                    return corrected_output
            except Exception as e:
                logger.error(f"Error during execution of sub-strategy {strategy.__class__.__name__}: {e}", exc_info=True)
        logger.warning("All sequential correction strategies failed.")
        return None


# Map of strategy names to their classes for dynamic instantiation
STRATEGY_MAP: Dict[str, Type[CorrectionStrategy]] = {
    "simple_reprompt": SimpleRepromptStrategy,
    "self_correction": SelfCorrectionStrategy,
    "human_in_the_loop": HumanInTheLoopStrategy,
    "sequential": SequentialCorrectionStrategy,
}


def _create_strategy_instance(strategy_name: str, llm_interface: LLMInterface, **kwargs) -> CorrectionStrategy:
    """
    Helper function to create a strategy instance given its name and parameters.
    Handles recursive instantiation for 'sequential' strategies.
    """
    strategy_cls = STRATEGY_MAP.get(strategy_name)
    if not strategy_cls:
        raise ValueError(f"Unknown correction strategy: '{strategy_name}'")

    if strategy_name == "sequential":
        # For SequentialCorrectionStrategy, the 'strategies' parameter in kwargs
        # needs to be passed as 'strategies_config' to its constructor.
        sub_strategies_configs = kwargs.pop('strategies', [])
        if not isinstance(sub_strategies_configs, list):
            raise TypeError("SequentialCorrectionStrategy 'strategies' parameter must be a list of strategy configurations (dictionaries).")
        
        # Instantiate SequentialCorrectionStrategy, which internally calls _create_strategy_instance for its sub-strategies
        return strategy_cls(llm_interface=llm_interface, strategies_config=sub_strategies_configs, **kwargs)
    else:
        # For other strategies, just pass the kwargs directly
        return strategy_cls(llm_interface=llm_interface, **kwargs)


def get_correction_strategy(strategy_config: Dict[str, Any], llm_interface: LLMInterface) -> CorrectionStrategy:
    """
    Factory function to get a correction strategy based on a configuration dictionary.

    Args:
        strategy_config: A dictionary defining the strategy, e.g.,
                         {"name": "self_correction", "max_retries": 2, "correction_prompt_template": "..."}
                         For a sequential strategy:
                         {"name": "sequential", "strategies": [
                             {"name": "simple_reprompt", "max_retries": 1},
                             {"name": "self_correction", "max_retries": 2}
                         ]}
        llm_interface: An instance of LLMInterface to be used by the strategy.

    Returns:
        An instantiated CorrectionStrategy object.

    Raises:
        ValueError: If the strategy configuration is invalid or the strategy name is unknown.
        TypeError: If configuration types are incorrect (e.g., 'strategies' is not a list for sequential).
    """
    if not isinstance(strategy_config, dict) or "name" not in strategy_config:
        raise ValueError("Invalid strategy configuration. Must be a dictionary with a 'name' key.")

    strategy_name = strategy_config["name"]
    # All other keys are parameters for the strategy
    params = {k: v for k, v in strategy_config.items() if k != "name"}

    return _create_strategy_instance(strategy_name, llm_interface, **params)

```