```python
import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple

# Assuming these are defined in monitor/core/types.py or base_detector.py
# For this single file implementation, they are included directly.

class ProblemReport:
    """
    A standardized report format for issues detected by the monitor.
    """
    def __init__(self, detector_name: str, problem_type: str, description: str, context: Dict[str, Any]):
        if not isinstance(detector_name, str) or not detector_name:
            raise ValueError("detector_name must be a non-empty string.")
        if not isinstance(problem_type, str) or not problem_type:
            raise ValueError("problem_type must be a non-empty string.")
        if not isinstance(description, str) or not description:
            raise ValueError("description must be a non-empty string.")
        if not isinstance(context, dict):
            raise ValueError("context must be a dictionary.")

        self.detector_name = detector_name
        self.problem_type = problem_type
        self.description = description
        self.context = context

    def __repr__(self):
        return (f"ProblemReport(detector_name='{self.detector_name}', "
                f"problem_type='{self.problem_type}', description='{self.description[:100]}...')")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ProblemReport to a dictionary."""
        return {
            "detector_name": self.detector_name,
            "problem_type": self.problem_type,
            "description": self.description,
            "context": self.context
        }


class BaseDetector:
    """
    Base interface for all detectors.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else {}
        self.detector_name = self.__class__.__name__

    def detect(self, state_history: List[Dict[str, Any]]) -> Optional[ProblemReport]:
        """
        Analyzes the agent's state history to detect problematic patterns.

        :param state_history: A list of dictionaries, each representing an observation
                              from the StateManager. Each dict should at least have
                              'timestamp' (float), 'step_type' (str), and 'content' (Any).
        :return: An optional ProblemReport if an issue is detected, otherwise None.
        """
        raise NotImplementedError


# Helper function to create a stable hash from content, robust to different types.
# This ensures that equivalent content (e.g., two dicts with same keys/values) produce the same hash.
def _hash_content(content: Any) -> str:
    """
    Creates a stable SHA256 hash for various content types, suitable for comparison.
    """
    if isinstance(content, (str, int, float, bool, type(None))):
        return hashlib.sha256(str(content).encode('utf-8')).hexdigest()
    elif isinstance(content, (dict, list)):
        try:
            # For dicts/lists, ensure stable JSON serialization before hashing
            # json.dumps with sort_keys=True guarantees consistent string representation
            return hashlib.sha256(json.dumps(content, sort_keys=True).encode('utf-8')).hexdigest()
        except TypeError:
            # Fallback if content is not JSON serializable (e.g., contains objects)
            return hashlib.sha256(repr(content).encode('utf-8')).hexdigest()
    else:
        # Fallback for other complex types
        return hashlib.sha256(repr(content).encode('utf-8')).hexdigest()


class ProgressDetector(BaseDetector):
    """
    Detects lack of progress by monitoring for stagnation in the agent's output
    or repetitive/unchanging internal agent states/actions over a threshold
    number of steps.
    """
    DEFAULT_CONFIG = {
        "min_observations_for_detection": 5,
        "output_stagnation_steps": 10,  # Number of total steps the agent's 'output' content can remain unchanged
        "state_stagnation_steps": 15,   # Number of agent-initiated steps (thought, tool_call) to observe for novelty
        "unique_state_threshold": 2,    # Max number of unique (thought/tool_call) actions allowed in stagnation window
        "stagnation_time_seconds": 60,  # Optional: Time in seconds for overall stagnation, for future use
        "agent_action_step_types": ["thought", "tool_call"], # Step types that represent agent's internal actions
        "output_step_type": "output"    # The step type representing the final output
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.config = {**self.DEFAULT_CONFIG, **self.config}
        self.detector_name = "ProgressDetector"

    def detect(self, state_history: List[Dict[str, Any]]) -> Optional[ProblemReport]:
        """
        Analyzes the agent's state history to detect lack of progress.
        Checks for:
        1. Stagnation in the agent's final output over a period.
        2. Repetitive or unchanging internal states/actions (thoughts, tool calls).
        """
        if not isinstance(state_history, list):
            raise TypeError("state_history must be a list.")
        
        if len(state_history) < self.config["min_observations_for_detection"]:
            return None # Not enough history to make a reliable detection

        # Ensure history is sorted by timestamp (StateManager should ideally guarantee this)
        try:
            state_history.sort(key=lambda x: x.get("timestamp", 0))
        except TypeError as e:
            # Handle cases where 'timestamp' might be missing or non-sortable
            # For a prototype, we'll proceed but log/warn in a real system.
            pass

        # 1. Check for Output Stagnation
        output_stagnation_report = self._check_output_stagnation(state_history)
        if output_stagnation_report:
            return output_stagnation_report

        # 2. Check for Internal State Stagnation (lack of novelty in thoughts/tool calls)
        state_stagnation_report = self._check_internal_state_stagnation(state_history)
        if state_stagnation_report:
            return state_stagnation_report

        return None

    def _check_output_stagnation(self, state_history: List[Dict[str, Any]]) -> Optional[ProblemReport]:
        """
        Checks if the agent's declared output has not changed for a significant number of steps.
        This considers the total number of steps in the state history.
        """
        current_step_count = len(state_history)

        # Get the latest output content
        latest_output_content = None
        latest_output_index = -1
        for i in range(current_step_count - 1, -1, -1):
            if state_history[i].get("step_type") == self.config["output_step_type"]:
                latest_output_content = state_history[i].get("content")
                latest_output_index = i
                break

        if latest_output_content is None:
            return None # No output has been generated yet

        # If not enough steps have passed since the latest output to evaluate stagnation
        if current_step_count - 1 - latest_output_index < self.config["output_stagnation_steps"]:
            return None

        # Determine the earliest point in history to compare against.
        # This is `output_stagnation_steps` ago from the latest observation.
        comparison_point_index = current_step_count - 1 - self.config["output_stagnation_steps"]

        # Find the output content that was active at or before the comparison_point_index
        output_at_comparison_point_content = None
        for i in range(comparison_point_index, -1, -1):
            if state_history[i].get("step_type") == self.config["output_step_type"]:
                output_at_comparison_point_content = state_history[i].get("content")
                break
        
        if output_at_comparison_point_content is None:
            return None # No previous output found for comparison within or before the window

        # If the latest output is different from the output at the comparison point, no stagnation
        if _hash_content(latest_output_content) != _hash_content(output_at_comparison_point_content):
            return None

        # Check for any change in output *between* the comparison point and the latest output
        # (inclusive of latest output, exclusive of the output_at_comparison_point_content's index)
        # This ensures we didn't just loop back to an old output after making progress
        for i in range(comparison_point_index + 1, current_step_count):
            obs = state_history[i]
            if obs.get("step_type") == self.config["output_step_type"]:
                if _hash_content(obs.get("content")) != _hash_content(latest_output_content):
                    return None # Output *did* change at some point within the window

        # If we reached here, the output has been consistent for at least `output_stagnation_steps`
        return ProblemReport(
            detector_name=self.detector_name,
            problem_type="OutputStagnation",
            description=f"Agent's output has not changed for at least "
                        f"{self.config['output_stagnation_steps']} steps. Current output: '{latest_output_content}'.",
            context={
                "latest_output": latest_output_content,
                "stagnated_steps_since_last_change": current_step_count - (output_at_comparison_point_content_index if output_at_comparison_point_content_index != -1 else 0),
                "threshold_steps": self.config["output_stagnation_steps"],
                "relevant_history_snippet": state_history[max(0, comparison_point_index):] # Show recent history
            }
        )

    def _check_internal_state_stagnation(self, state_history: List[Dict[str, Any]]) -> Optional[ProblemReport]:
        """
        Checks if the agent's internal actions (thoughts, tool calls) are repetitive
        or show a lack of new content over a recent window of steps.
        """
        # Filter for agent-initiated actions defined in config
        agent_actions_history = [
            obs for obs in state_history
            if obs.get("step_type") in self.config["agent_action_step_types"]
        ]

        if len(agent_actions_history) < self.config["state_stagnation_steps"]:
            return None # Not enough agent actions to evaluate stagnation

        # Consider only the most recent 'state_stagnation_steps' agent actions
        recent_agent_actions = agent_actions_history[-self.config["state_stagnation_steps"]:]

        unique_action_hashes = set()
        for obs in recent_agent_actions:
            # Create a tuple of (step_type, hashed_content) for robust uniqueness check
            action_identifier = (obs.get("step_type"), _hash_content(obs.get("content")))
            unique_action_hashes.add(action_identifier)

        if len(unique_action_hashes) <= self.config["unique_state_threshold"]:
            # Optional time-based check for future refinement, currently not strictly enforced
            # to keep step-based detection primary for the prototype.
            # first_obs_time = recent_agent_actions[0].get("timestamp", 0)
            # last_obs_time = recent_agent_actions[-1].get("timestamp", time.time())
            # time_elapsed = last_obs_time - first_obs_time
            # if time_elapsed < self.config.get("stagnation_time_seconds", 0):
            #     return None # Not enough real time has passed for this to be a "stall" over time

            return ProblemReport(
                detector_name=self.detector_name,
                problem_type="InternalStateStagnation",
                description=f"Agent's internal state (thoughts, tool calls) has shown "
                            f"only {len(unique_action_hashes)} unique actions "
                            f"over the last {len(recent_agent_actions)} agent-initiated steps "
                            f"(threshold: {self.config['unique_state_threshold']} unique actions).",
                context={
                    "unique_actions_in_window": len(unique_action_hashes),
                    "relevant_steps_in_window": len(recent_agent_actions),
                    "threshold_unique_actions": self.config["unique_state_threshold"],
                    "window_start_timestamp": recent_agent_actions[0].get("timestamp"),
                    "window_end_timestamp": recent_agent_actions[-1].get("timestamp"),
                    "recent_agent_actions_snippet": recent_agent_actions
                }
            )
        return None
```