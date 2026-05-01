```python
import os

class RLGuardConfig:
    """
    Centralized configuration for the RLGuard system and the simulated RL environment.
    """

    # --- Global Settings ---
    DEBUG_MODE = True
    # Assumes config.py is located in the 'rlguard' directory at the project root level.
    # e.g., /project_root/rlguard/config.py
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


    # --- RL Environment Simulation Configuration ---
    RL_ENV = {
        "EPISODES": 500,
        "STEPS_PER_EPISODE": 100,
        "DISCOUNT_FACTOR": 0.99,
        "LLM_MODEL_NAME": "LLM_Exploration_Simulator_v1",
        "ACTION_SPACE_SIZE": 10,  # Example: number of discrete actions LLM can choose
        "OBSERVATION_SPACE_SIZE": 20,  # Example: dimension of state vector
        "REWARD_SCALING_FACTOR": 1.0,
        "INITIAL_EXPLORATION_EPSILON": 0.1,  # Initial epsilon for epsilon-greedy policy
    }

    # --- Telemetry Configuration ---
    TELEMETRY = {
        "ENABLED": True,
        "METRICS_TO_COLLECT": [
            "action_distribution_entropy",
            "novelty_score",
            "reward_variance",
            "trajectory_diversity",
            "subgoal_completion_rate",
            "kl_divergence_from_baseline_policy", # Additional common RL metric
        ],
        "COLLECTION_INTERVAL": "step",  # Options: "step", "episode", "batch"
        "STREAMING_BATCH_SIZE": 10,     # How many metric observations to batch before sending to engine
        "METRIC_WINDOW_SIZE_FOR_CALC": 10, # For metrics like rolling reward variance or novelty
    }

    # --- Anomaly Detection Configuration ---
    ANOMALY_DETECTION = {
        "ENABLED": True,
        "INITIAL_BASELINE_EPISODES": 20, # Number of initial episodes to establish a baseline before detection starts
        "DETECTORS": {
            "STATISTICAL_DETECTOR": {
                "ENABLED": True,
                "Z_SCORE_THRESHOLD": 3.0, # Flag anomalies if Z-score exceeds this for any metric
                "METRIC_SPECIFIC_THRESHOLDS": {
                    # Example: Define specific min/max thresholds for individual metrics if needed
                    "action_distribution_entropy": {"min": 0.5, "max": 2.5},
                    "reward_variance": {"min": 0.0, "max": 100.0},
                },
                "ROLLING_WINDOW_SIZE": 50, # Window for calculating rolling mean/stddev for Z-score
            },
            "ML_BASED_DETECTOR": {
                "ENABLED": False, # Set to True to enable ML-based detection
                "MODEL_PATH": os.path.join(PROJECT_ROOT, "rlguard", "anomaly_detection", "models", "ml_detector_model.pkl"),
                "RETRAIN_INTERVAL_EPISODES": 100, # How often to retrain the ML model
                "ONLINE_LEARNING_ENABLED": False, # If true, model updates incrementally
                "CLASSIFIER_THRESHOLD": 0.7, # Probability threshold for anomaly classification
            },
        },
        "BASELINE_MANAGER": {
            "UPDATE_STRATEGY": "rolling_window", # Options: "episodic", "rolling_window", "fixed"
            "BASELINE_WINDOW_SIZE": 100, # Number of data points (steps/episodes) to consider for baseline
            "BASELINE_SAVE_PATH": os.path.join(PROJECT_ROOT, "rlguard", "anomaly_detection", "baselines", "current_baselines.json"),
        }
    }

    # --- Dynamic Intervention Configuration ---
    INTERVENTION = {
        "ENABLED": True,
        "HUMAN_ALERT_THRESHOLD": "CRITICAL", # Anomaly severity that triggers a human alert
        "DEFAULT_INTERVENTION_STRATEGY": "ALERT_HUMAN", # Fallback if no specific policy matches
        "INTERVENTION_POLICIES": {
            # Map anomaly types/severities to strategies and their parameters
            "EXPLORATION_HACKING_HIGH": {
                "strategy": "ADJUST_EXPLORATION_HYPERPARAMETERS",
                "params": {
                    "hyperparameter": "epsilon",
                    "adjustment_type": "increase", # "increase", "decrease", "set"
                    "value": 0.2, # Increase epsilon by 0.2
                    "max_value": 0.9,
                    "duration_episodes": 5, # Apply for 5 episodes
                }
            },
            "REWARD_SUBVERSION_MEDIUM": {
                "strategy": "APPLY_REWARD_PENALTY",
                "params": {
                    "penalty_magnitude": -50.0,
                    "duration_steps": 20, # Apply penalty for 20 steps
                    "target_action_space": [], # Optional: only penalize specific actions. Empty list means all actions.
                }
            },
            "NOVELTY_DRASTIC_DROP_LOW": {
                "strategy": "DIRECTED_EXPLORATION",
                "params": {
                    "prompt_template": "Your current exploration is too narrow. Try focusing on the following unexplored areas: {unexplored_hints}.",
                    "hint_generator_function_name": "generate_novel_area_hints", # A function name from strategies.py to call to get hints
                    "duration_episodes": 3,
                }
            },
            "ALIGNMENT_DEVIATION_CRITICAL": {
                "strategy": "SWITCH_TO_SUPERVISED_FINE_TUNING",
                "params": {
                    "sft_dataset_id": "safety_alignment_dataset_v3",
                    "duration_steps": 50,
                    "model_weights_path": os.path.join(PROJECT_ROOT, "sft_models", "aligned_weights_v3.pt"),
                }
            },
            "UNKNOWN_ANOMALY_DEFAULT": { # General fallback for any unhandled anomaly type
                "strategy": "ALERT_HUMAN",
                "params": {
                    "severity": "WARNING",
                    "message": "An unhandled anomaly type was detected. Review logs."
                }
            }
        }
    }

    # --- Observability & Audit Trails Configuration ---
    OBSERVABILITY = {
        "LOGGING_LEVEL": "INFO", # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        "LOG_FILE_PATH": os.path.join(PROJECT_ROOT, "logs", "rlguard_training.log"),
        "AUDIT_TRAIL_FILE_PATH": os.path.join(PROJECT_ROOT, "logs", "rlguard_audit_trail.json"),
        "DASHBOARD_ENABLED": False, # Placeholder, actual implementation would involve a separate service
        "DASHBOARD_REFRESH_RATE_SECONDS": 5,
    }

    # --- Error Handling Configuration (for RLGuard's internal operations) ---
    ERROR_HANDLING = {
        "LOG_EXCEPTIONS": True,
        "RETRY_ATTEMPTS_ON_FAILURE": 3,
        "RETRY_DELAY_SECONDS": 1.0,
        "CRITICAL_FAILURE_ACTIONS": ["STOP_TRAINING", "NOTIFY_OPS_TEAM"], # Example: "STOP_TRAINING", "NOTIFY_PAGERDUTY"
    }

    @classmethod
    def initialize(cls):
        """
        Ensures necessary directories for logs, baselines, and models exist.
        This helps prevent FileNotFoundError at runtime.
        """
        try:
            # Log directory
            log_dir = os.path.dirname(cls.OBSERVABILITY["LOG_FILE_PATH"])
            os.makedirs(log_dir, exist_ok=True)

            # Baseline directory
            baseline_dir = os.path.dirname(cls.ANOMALY_DETECTION["BASELINE_MANAGER"]["BASELINE_SAVE_PATH"])
            os.makedirs(baseline_dir, exist_ok=True)

            # ML model directory (if ML detector is enabled)
            ml_model_dir = os.path.dirname(cls.ANOMALY_DETECTION["DETECTORS"]["ML_BASED_DETECTOR"]["MODEL_PATH"])
            os.makedirs(ml_model_dir, exist_ok=True)

            # SFT models directory (if SFT intervention is enabled/configured)
            sft_policy = cls.INTERVENTION["INTERVENTION_POLICIES"].get("ALIGNMENT_DEVIATION_CRITICAL", {})
            if sft_policy.get("strategy") == "SWITCH_TO_SUPERVISED_FINE_TUNING" and "model_weights_path" in sft_policy.get("params", {}):
                sft_model_dir = os.path.dirname(sft_policy["params"]["model_weights_path"])
                os.makedirs(sft_model_dir, exist_ok=True)

        except Exception as e:
            # Basic error handling for directory creation
            print(f"Error initializing RLGuard configuration directories: {e}")
            # In a real system, you might log this more robustly or raise a critical error.

```