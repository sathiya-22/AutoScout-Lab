```python
import collections
from typing import List, Any, Optional, Tuple

# Assuming these are defined in the project structure as per architecture notes
from monitor.detectors.base_detector import BaseDetector, ProblemReport
from monitor.state_manager import Observation # Assuming Observation class is defined here

class LoopDetector(BaseDetector):
    """
    A detector for identifying repetitive loops in an AI agent's behavior.

    This detector looks for sequences of agent observations (thoughts, tool calls, etc.)
    that repeat consecutively within a defined historical window.
    """

    DETECTOR_ID = "loop_detector"

    def __init__(self,
                 window_size: int = 20,
                 min_repetitions: int = 3,
                 min_sequence_length: int = 2,
                 observation_types_to_monitor: Optional[List[str]] = None):
        """
        Initializes the LoopDetector with configuration parameters.

        Args:
            window_size (int): The maximum number of recent relevant observations to consider
                                for loop detection. This window effectively defines the maximum
                                total length of (repeated sequence * repetitions) that can be detected.
            min_repetitions (int): The minimum number of times a sequence must repeat
                                   consecutively to be flagged as a loop. Must be at least 2.
            min_sequence_length (int): The minimum length of a sequence of observations
                                       to be considered for loop detection. Must be at least 1.
            observation_types_to_monitor (Optional[List[str]]): A list of observation
                                                                 `step_type` values to
                                                                 include in loop analysis.
                                                                 If None, defaults to ["thought", "tool_call"].
        Raises:
            ValueError: If configuration parameters are invalid.
        """
        super().__init__(self.DETECTOR_ID)

        if min_sequence_length < 1:
            raise ValueError("min_sequence_length must be at least 1.")
        if min_repetitions < 2:
            raise ValueError("min_repetitions must be at least 2 for a 'loop' to be detected.")
        if window_size < min_sequence_length * min_repetitions:
            # Ensure there's enough history in the window to potentially find a loop
            # E.g., if min_sequence_length=2 and min_repetitions=3, window_size needs to be at least 6
            # to see 'A-B-A-B-A-B'.
            raise ValueError(f"window_size ({window_size}) must be at least "
                             f"min_sequence_length * min_repetitions "
                             f"({min_sequence_length * min_repetitions}) to enable detection.")

        self.window_size = window_size
        self.min_repetitions = min_repetitions
        self.min_sequence_length = min_sequence_length
        self.observation_types_to_monitor = observation_types_to_monitor or [
            "thought", "tool_call"
        ]

    def _get_comparable_observation_data(self, obs: Observation) -> Tuple[str, Any]:
        """
        Converts an Observation object into a hashable and comparable representation.
        This is crucial for robust sequence comparison.

        Args:
            obs (Observation): The observation object to process.

        Returns:
            Tuple[str, Any]: A tuple representing the observation for comparison.
                             The first element is the step_type, subsequent elements
                             are normalized content data.
        """
        # Basic validation to ensure obs has expected attributes
        if not isinstance(obs, Observation) or not hasattr(obs, 'step_type') or not hasattr(obs, 'content'):
            # Fallback for unexpected observation formats or non-Observation objects
            return (f"UNKNOWN_TYPE_{type(obs).__name__}", str(obs))

        # Handle specific observation types for more precise comparison
        if obs.step_type == "tool_call":
            # Assuming obs.content is a dictionary like {"tool_name": "...", "tool_args": {...}}
            tool_name = obs.content.get("tool_name", "UNKNOWN_TOOL")
            tool_args = obs.content.get("tool_args", {})
            
            # Ensure tool_args are sorted for consistent hash/comparison if it's a dict.
            # This is important as dictionary iteration order might vary in older Python versions
            # and could lead to inconsistent tuple representations for the same logical arguments.
            if isinstance(tool_args, dict):
                comparable_args = tuple(sorted(tool_args.items()))
            else:
                # Fallback for non-dictionary tool_args
                comparable_args = str(tool_args)
            return (obs.step_type, tool_name, comparable_args)
        
        elif obs.step_type == "thought":
            # Assuming obs.content is a string
            return (obs.step_type, obs.content)
            
        elif obs.step_type == "tool_result":
            # For tool results, the exact content might be important for loop detection.
            # If results can be very large or highly variable, one might consider hashing
            # or summarizing them instead of using the full string representation for performance
            # or to detect 'semantic' loops despite minor content variations.
            # For this prototype, a string conversion is simple and effective.
            return (obs.step_type, str(obs.content))
        
        elif obs.step_type == "output":
            # For final output, typically less relevant for internal agent loops,
            # but included for completeness.
            return (obs.step_type, str(obs.content))

        else:
            # Default fallback for any other observation type, using its type and string representation of content.
            return (obs.step_type, str(obs.content))

    def detect(self, state_history: List[Observation]) -> Optional[ProblemReport]:
        """
        Detects repetitive loops in the agent's state history.

        Args:
            state_history (List[Observation]): A chronological list of agent observations.

        Returns:
            Optional[ProblemReport]: A ProblemReport if a loop is detected, otherwise None.
        """
        if not state_history:
            return None

        # 1. Filter and process observations to only include relevant types
        # and convert them into a consistent, comparable format.
        processed_history = [
            self._get_comparable_observation_data(obs)
            for obs in state_history
            if obs.step_type in self.observation_types_to_monitor
        ]

        # 2. Consider only the most recent 'window_size' relevant observations.
        # This limits the scope of loop detection to recent activity, improving performance
        # and focusing on current problematic behavior.
        recent_relevant_observations = processed_history[-self.window_size:]

        # 3. Check if there's enough history in the window to even form a minimal loop
        # (i.e., min_sequence_length repeated min_repetitions times).
        if len(recent_relevant_observations) < self.min_sequence_length * self.min_repetitions:
            return None

        # 4. Iterate through possible sequence lengths.
        # We look for sequences starting from `min_sequence_length` up to the maximum
        # possible length that could repeat `min_repetitions` times within our window.
        max_possible_sequence_len = len(recent_relevant_observations) // self.min_repetitions

        for sequence_length in range(self.min_sequence_length, max_possible_sequence_len + 1):
            if sequence_length == 0: 
                continue # Should be caught by min_sequence_length check, but good for robustness

            # Define the 'current' (most recent) sequence of this length.
            # This is the pattern we are attempting to identify if it repeats.
            current_sequence = tuple(recent_relevant_observations[-sequence_length:])

            repetitions = 1 # The current_sequence itself counts as one repetition

            # Look backwards in the `recent_relevant_observations` for consecutive repetitions
            # of `current_sequence`.
            # We start checking from the segment immediately preceding `current_sequence`.
            check_idx_start = len(recent_relevant_observations) - (2 * sequence_length)

            while check_idx_start >= 0:
                # Extract the preceding sequence of the same length
                prev_sequence = tuple(recent_relevant_observations[check_idx_start : check_idx_start + sequence_length])

                if prev_sequence == current_sequence:
                    repetitions += 1
                    # Move further back in history to find the next potential repetition
                    check_idx_start -= sequence_length
                else:
                    # The immediate consecutive repetition pattern is broken, so stop searching
                    break
            
            # If the required number of repetitions is met, a loop is detected.
            if repetitions >= self.min_repetitions:
                problem_description = (
                    f"Agent detected in a repetitive loop. "
                    f"A sequence of {sequence_length} observations "
                    f"has repeated {repetitions} times consecutively in recent history."
                )
                # Return a ProblemReport with details about the detected loop.
                return ProblemReport(
                    detector_id=self.DETECTOR_ID,
                    description=problem_description,
                    severity="high",
                    data={
                        "sequence_length": sequence_length,
                        "repetitions": repetitions,
                        # Calculate the start and end indices of the *entire* detected pattern
                        # within the `recent_relevant_observations` list.
                        "pattern_start_index_in_recent_history": len(recent_relevant_observations) - (repetitions * sequence_length),
                        "pattern_end_index_in_recent_history": len(recent_relevant_observations) - 1,
                        # Provide a string representation of the detected pattern sample for debugging.
                        "detected_pattern_sample": str(current_sequence) 
                    }
                )

        # If the loop finishes without finding any repeating patterns meeting the criteria,
        # then no loop is detected in the current history.
        return None
```