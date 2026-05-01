```python
import numpy as np
import random
import math

class LLMPolicy:
    def __init__(self, action_space_size: int, config: dict = None, seed: int = None):
        """
        Initializes the LLM Policy.
        Simulates an LLM's action selection in an RL environment.

        Args:
            action_space_size (int): The size of the discrete action space.
            config (dict): Configuration dictionary for the LLM policy,
                           e.g., initial temperature, epsilon for exploration.
            seed (int, optional): Seed for random number generators to ensure reproducibility.
        """
        if action_space_size <= 0:
            raise ValueError("action_space_size must be a positive integer.")

        self.action_space_size = action_space_size
        self.config = config if config is not None else {}

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.seed = seed

        # Default exploration parameters
        self.temperature = float(self.config.get('initial_temperature', 1.0))
        self.epsilon = float(self.config.get('initial_epsilon', 0.1)) # For epsilon-greedy fallback/hybrid
        # Ensure temperature is positive
        if self.temperature <= 0:
            print(f"Warning: Initial temperature was {self.temperature}. Clamping to 0.1.")
            self.temperature = 0.1
        # Ensure epsilon is within [0, 1]
        self.epsilon = max(0.0, min(1.0, self.epsilon))


        self.base_prompt_template = self.config.get(
            'base_prompt_template',
            "You are an agent in a game. Current situation: {state_description}. Available actions: {actions_list}. What is your next move?"
        )
        self.current_prompt = None # Will be set during act()
        self.last_action_logits = None
        self.last_action_probs = None
        self.last_action = None
        self.internal_state = {} # Placeholder for more complex LLM internal memory/state

        # For simulated 'sub-goal completion' - purely illustrative here
        self.sub_goals = self.config.get('sub_goals', [])
        self.completed_sub_goals = set()

    def _simulate_llm_response(self, state: dict, prompt: str) -> np.ndarray:
        """
        Simulates an LLM generating logits for actions based on state and prompt.
        In a real scenario, this would involve calling a true LLM API.
        Here, we generate random logits, potentially biased by some 'learned' behavior
        or internal state directives.
        """
        # A simple simulation: logits are random with a slight state dependency
        # if the state has a 'vector_representation' or a 'value'
        state_influence = 0.0
        if 'vector_representation' in state and isinstance(state['vector_representation'], list) and state['vector_representation']:
            # Take the average or sum of the state vector as an influence
            state_influence = np.mean(state['vector_representation'])
        elif 'value' in state and isinstance(state['value'], (int, float)):
            state_influence = state['value']

        # Start with random logits, scaled
        logits = np.random.rand(self.action_space_size) * 2 - 1 # between -1 and 1

        # Apply a general state influence
        logits += state_influence * 0.1 # Small influence

        # Example bias from internal state (e.g., set by RLGuard for targeted exploration)
        if self.internal_state.get('goal_seeking', False) and 'target_action' in self.internal_state:
            target_action = self.internal_state['target_action']
            if 0 <= target_action < self.action_space_size:
                logits[target_action] += self.internal_state.get('target_action_bias', 3.0) # Stronger bias

        return logits

    def act(self, state: dict, prompt: str = None) -> (int, dict):
        """
        Determines the next action based on the current state using the simulated LLM policy.

        Args:
            state (dict): The current state of the environment. Expected to have a 'description' key.
            prompt (str, optional): An optional prompt to override or supplement the base prompt.

        Returns:
            tuple: (action (int), info (dict))
                   action: The chosen action.
                   info: Dictionary containing additional info like action probabilities,
                         raw logits, current exploration parameters.
        """
        # Prepare state description for prompt
        state_description = state.get('description', str(state))
        actions_list_str = ', '.join(map(str, range(self.action_space_size)))

        # Format the current prompt for the LLM based on state
        current_step_prompt = prompt if prompt else self.base_prompt_template.format(
            state_description=state_description,
            actions_list=actions_list_str
        )
        self.current_prompt = current_step_prompt # Store for telemetry/logging

        # Simulate LLM generating action logits
        raw_logits = self._simulate_llm_response(state, current_step_prompt)
        self.last_action_logits = raw_logits

        # Apply temperature for sampling
        # Use softmax to convert logits to probabilities
        try:
            # Shift logits for numerical stability before exponentiation
            stable_logits = (raw_logits - np.max(raw_logits)) / self.temperature
            exp_logits = np.exp(stable_logits)
            sum_exp_logits = np.sum(exp_logits)

            if sum_exp_logits == 0: # All exp_logits were ~0, or temperature was extremely low/high causing overflow/underflow
                action_probabilities = np.ones(self.action_space_size) / self.action_space_size
                # print("Warning: Sum of exponentiated logits was zero. Falling back to uniform probabilities.")
            else:
                action_probabilities = exp_logits / sum_exp_logits

            # Final check for NaNs in case of very weird numerical issues
            if np.isnan(action_probabilities).any():
                action_probabilities = np.ones(self.action_space_size) / self.action_space_size
                print("Warning: NaN detected in action probabilities. Falling back to uniform probabilities.")

        except Exception as e:
            print(f"Error applying temperature/softmax: {e}. Falling back to uniform probabilities.")
            action_probabilities = np.ones(self.action_space_size) / self.action_space_size

        self.last_action_probs = action_probabilities

        # Epsilon-greedy exploration blend (or pure sampling if epsilon is 0)
        if random.random() < self.epsilon:
            action = random.randrange(self.action_space_size) # Random action
        else:
            # If probabilities don't sum to 1 (due to floating point errors), normalize them
            if not np.isclose(np.sum(action_probabilities), 1.0):
                action_probabilities = action_probabilities / np.sum(action_probabilities)
            action = np.random.choice(self.action_space_size, p=action_probabilities) # Sample based on probs

        self.last_action = action

        # Simulate sub-goal completion based on state/action for telemetry
        self._update_sub_goals(state, action)

        info = {
            'action_probabilities': self.last_action_probs.tolist(), # Convert to list for serialization
            'raw_logits': self.last_action_logits.tolist(),
            'chosen_action': self.last_action,
            'temperature': self.temperature,
            'epsilon': self.epsilon,
            'prompt_used': self.current_prompt,
            'sub_goal_completion_rate': self.get_sub_goal_completion_rate(), # Include in info for convenience
            'internal_state': self.internal_state.copy() # Copy to avoid external modification
        }
        return self.last_action, info

    def _update_sub_goals(self, state: dict, action: int):
        """
        Simulates checking for sub-goal completion based on current state and action.
        This is a highly simplified placeholder.
        Sub-goal format: {'id': 'goal1', 'condition': {'state_key': 'value', 'action': N}}
        """
        if not self.sub_goals:
            return

        for sub_goal_def in self.sub_goals:
            goal_id = sub_goal_def.get('id')
            if not goal_id or goal_id in self.completed_sub_goals:
                continue

            condition_met = True
            conditions = sub_goal_def.get('condition', {})

            # Check state conditions
            if 'state_key' in conditions and 'state_value' in conditions:
                if state.get(conditions['state_key']) != conditions['state_value']:
                    condition_met = False
            elif 'state_contains' in conditions and isinstance(state.get('description'), str):
                if conditions['state_contains'] not in state['description']:
                    condition_met = False

            # Check action condition
            if condition_met and 'action' in conditions:
                if action != conditions['action']:
                    condition_met = False

            if condition_met:
                self.completed_sub_goals.add(goal_id)

    # --- Telemetry Methods ---
    def get_action_distribution_entropy(self) -> float:
        """Calculates the entropy of the last action probability distribution."""
        if self.last_action_probs is None:
            return 0.0 # No action taken yet

        # Ensure probabilities sum to 1 and handle log(0)
        probs = np.array(self.last_action_probs)
        if np.sum(probs) == 0:
            return 0.0 # No valid probabilities

        probs = probs / np.sum(probs)
        # Filter out zero probabilities to avoid log(0) warnings/errors
        filtered_probs = probs[probs > 1e-9] # Small epsilon to include near-zero values
        entropy = -np.sum(filtered_probs * np.log(filtered_probs))
        return entropy

    def get_last_action_logits(self) -> np.ndarray:
        """Returns the raw logits for the last action selection."""
        return self.last_action_logits if self.last_action_logits is not None else np.array([])

    def get_current_exploration_params(self) -> dict:
        """Returns the current exploration parameters."""
        return {'temperature': self.temperature, 'epsilon': self.epsilon}

    def get_sub_goal_completion_rate(self) -> float:
        """Returns the rate of sub-goal completion."""
        if not self.sub_goals:
            return 1.0 # No sub-goals defined, so consider 100% completion
        if not self.completed_sub_goals:
            return 0.0
        return len(self.completed_sub_goals) / len(self.sub_goals)

    # --- Intervention Methods ---
    def adjust_exploration_params(self, temperature: float = None, epsilon: float = None):
        """
        Adjusts the policy's exploration parameters.

        Args:
            temperature (float, optional): New temperature for softmax sampling.
            epsilon (float, optional): New epsilon for epsilon-greedy exploration.
        """
        if temperature is not None:
            if temperature > 0:
                self.temperature = float(temperature)
            else:
                print(f"Warning: Attempted to set temperature <= 0 ({temperature}). Keeping current temperature: {self.temperature}.")
        if epsilon is not None:
            if 0 <= epsilon <= 1:
                self.epsilon = float(epsilon)
            else:
                print(f"Warning: Attempted to set epsilon outside [0, 1] ({epsilon}). Keeping current epsilon: {self.epsilon}.")

    def update_prompt_template(self, new_template: str):
        """
        Updates the base prompt template used by the LLM.
        """
        if new_template and isinstance(new_template, str):
            self.base_prompt_template = new_template
        else:
            print(f"Warning: Attempted to set an invalid prompt template ({new_template}). Keeping current.")

    def set_internal_state(self, key: str, value: any):
        """
        Sets or updates a key-value pair in the LLM's internal state/memory.
        This could be used to guide behavior, e.g., 'set_internal_state('goal_seeking', True)'.
        """
        if key is not None:
            self.internal_state[key] = value
        else:
            print("Warning: Attempted to set internal state with a None key.")

    def reset_internal_state(self):
        """Resets the LLM's internal state and completed sub-goals."""
        self.internal_state = {}
        self.completed_sub_goals = set()
```