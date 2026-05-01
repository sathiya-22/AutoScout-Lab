```python
import numpy as np
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class TelemetryCollector:
    """
    Responsible for instrumenting the LLM policy and RL environment to extract diverse
    exploration metrics and raw data points during RL training.
    """

    def __init__(self, llm_policy, rl_environment, reward_history_window: int = 100):
        """
        Initializes the TelemetryCollector with references to the LLM policy and RL environment.

        Args:
            llm_policy: The LLM policy object, expected to have methods/attributes
                        to expose action probabilities, logits, etc.
            rl_environment: The RL environment object, expected to provide state, reward, info.
            reward_history_window (int): The number of recent rewards to keep for basic variance calculation.
        """
        if llm_policy is None:
            raise ValueError("LLM Policy object cannot be None.")
        if rl_environment is None:
            raise ValueError("RL Environment object cannot be None.")
        if not isinstance(reward_history_window, int) or reward_history_window <= 0:
            raise ValueError("reward_history_window must be a positive integer.")

        self.llm_policy = llm_policy
        self.rl_environment = rl_environment
        self.reward_history = deque(maxlen=reward_history_window)
        
        logger.info(f"TelemetryCollector initialized with reward history window: {reward_history_window}")

    def _calculate_action_entropy(self, action_probabilities: np.ndarray) -> float:
        """
        Calculates the entropy of the action distribution.
        
        Args:
            action_probabilities (np.ndarray): A 1D numpy array of probabilities for each action.
        
        Returns:
            float: The entropy of the distribution. Returns 0 if probabilities are invalid or empty.
        """
        if action_probabilities is None or not isinstance(action_probabilities, np.ndarray) or action_probabilities.size == 0:
            logger.debug("Attempted to calculate entropy with invalid or empty action probabilities.")
            return 0.0
        
        # Filter out zero probabilities to avoid log(0)
        non_zero_probs = action_probabilities[action_probabilities > 0]
        if non_zero_probs.size == 0:
            return 0.0 # All probabilities are zero, or array was invalid.
            
        entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))
        return float(entropy)

    def collect_step_metrics(self,
                             state: np.ndarray,
                             action: int,
                             reward: float,
                             next_state: np.ndarray,
                             done: bool,
                             info: dict,
                             llm_policy_output: dict = None) -> dict:
        """
        Collects a dictionary of raw data and basic exploration metrics for the current RL step.

        Args:
            state (np.ndarray): The observation from the environment before the action.
            action (int): The action taken by the LLM policy.
            reward (float): The reward received from the environment.
            next_state (np.ndarray): The observation from the environment after the action.
            done (bool): Whether the episode has ended.
            info (dict): Auxiliary information from the environment.
            llm_policy_output (dict, optional): Dictionary containing policy-specific outputs
                                                like 'action_probabilities', 'action_logits',
                                                'internal_state_embedding'. Defaults to None.

        Returns:
            dict: A dictionary containing collected raw data and computed metrics for the step.
        """
        if not isinstance(state, np.ndarray):
            logger.warning(f"State is not a numpy array: {type(state)}. Attempting to convert or use directly.")
        if not isinstance(next_state, np.ndarray):
            logger.warning(f"Next state is not a numpy array: {type(next_state)}. Attempting to convert or use directly.")

        collected_data = {
            "timestamp": time.time(),
            "current_state": state.tolist() if isinstance(state, np.ndarray) else state,
            "action_taken": action,
            "reward_received": reward,
            "next_state": next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            "episode_done": done,
            "environment_info": info,
        }

        self.reward_history.append(reward)
        collected_data["recent_reward_mean"] = float(np.mean(self.reward_history)) if self.reward_history else 0.0
        collected_data["recent_reward_std"] = float(np.std(self.reward_history)) if len(self.reward_history) > 1 else 0.0

        if llm_policy_output and isinstance(llm_policy_output, dict):
            action_probs = llm_policy_output.get("action_probabilities")
            action_logits = llm_policy_output.get("action_logits")
            internal_embedding = llm_policy_output.get("internal_state_embedding")

            if action_probs is not None:
                try:
                    action_entropy = self._calculate_action_entropy(np.array(action_probs))
                    collected_data["action_distribution_entropy"] = action_entropy
                    collected_data["action_probabilities"] = action_probs.tolist() if isinstance(action_probs, np.ndarray) else action_probs
                except Exception as e:
                    logger.error(f"Error processing action probabilities for entropy: {e}")
                    collected_data["action_distribution_entropy"] = None
                    collected_data["action_probabilities"] = None
            else:
                collected_data["action_distribution_entropy"] = None
                collected_data["action_probabilities"] = None

            if action_logits is not None:
                collected_data["action_logits"] = action_logits.tolist() if isinstance(action_logits, np.ndarray) else action_logits
            else:
                collected_data["action_logits"] = None

            if internal_embedding is not None:
                collected_data["llm_internal_embedding"] = internal_embedding.tolist() if isinstance(internal_embedding, np.ndarray) else internal_embedding
            else:
                collected_data["llm_internal_embedding"] = None
        else:
            logger.debug("No valid LLM policy output provided for this step.")
            collected_data["action_distribution_entropy"] = None
            collected_data["action_probabilities"] = None
            collected_data["action_logits"] = None
            collected_data["llm_internal_embedding"] = None

        # Sub-goal completion from environment info
        sub_goal_reached = info.get("sub_goal_reached", False)
        collected_data["sub_goal_completion"] = bool(sub_goal_reached)
        
        # Placeholders for metrics typically computed over longer sequences or history by other modules
        collected_data["novelty_score"] = None
        collected_data["trajectory_diversity_metric"] = None
        
        return collected_data

```