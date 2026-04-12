"""
Configuration settings for the AI agent monitoring framework.
This file centralizes parameters for agents, monitor core, instrumentation, state manager,
detectors, and interventions, allowing for easy adjustment without code changes.
"""

# --- General Settings ---
DEBUG_MODE: bool = True
LOG_LEVEL: str = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- Agent Settings (for Monitored Agent) ---
AGENT_MODEL_NAME: str = "gpt-4o"  # Default LLM model for agents
AGENT_TEMPERATURE: float = 0.7     # Default LLM temperature for agents
AGENT_MAX_REPLAN_ATTEMPTS: int = 3 # Max times an agent can be asked to replan for a single issue
AGENT_SIMULATED_FAILURE_RATE: float = 0.3 # Probability for demo_agent to simulate a stall/loop (0.0 to 1.0)
AGENT_SIMULATED_FAILURE_TRIGGER_STEP: int = 5 # Step at which demo_agent might start simulating a failure

# --- Monitor Core Settings ---
MONITOR_CHECK_INTERVAL_STEPS: int = 1 # How often the monitor checks detectors (e.g., every N agent steps)
MONITOR_CHECK_INTERVAL_SECONDS: float = 0.1 # Max delay between checks if agent is fast

# --- State Manager Settings ---
MAX_STATE_HISTORY: int = 100 # Maximum number of historical observations to store

# --- Instrumentation Settings ---
# No specific configurable parameters for basic instrumentation beyond what's captured
# but this section could grow if instrumentation becomes more complex (e.g., selective logging)

# --- Detector Settings ---

# Base Detector Settings
PROBLEM_REPORT_EXPIRATION_SECONDS: int = 300 # How long a problem report remains active if not resolved

# Loop Detector
LOOP_DETECTOR_ENABLED: bool = True
LOOP_DETECTOR_WINDOW_SIZE: int = 5      # How many recent observations to consider for loop detection (min 2)
LOOP_DETECTOR_MIN_REPETITIONS: int = 2  # How many times a pattern must repeat to be a loop
# Types of observations to consider for looping patterns. Options: 'thought', 'tool_call', 'output', 'environment_feedback'
LOOP_DETECTOR_MONITOR_TYPES: list[str] = ["thought", "tool_call"]
LOOP_DETECTOR_SIMILARITY_THRESHOLD: float = 0.9 # For fuzzy string matching in detecting similar states (0.0 to 1.0)

# Progress Detector
PROGRESS_DETECTOR_ENABLED: bool = True
PROGRESS_DETECTOR_THRESHOLD_STEPS: int = 10 # Max steps without significant change
PROGRESS_DETECTOR_THRESHOLD_SECONDS: float = 60.0 # Max time (seconds) without significant change
PROGRESS_DETECTOR_MIN_OBSERVATIONS: int = 3 # Minimum observations before progress detection kicks in
# Types of observations whose lack of change signifies a stall. Options: 'thought', 'tool_call', 'output', 'environment_feedback'
PROGRESS_DETECTOR_MONITOR_TYPES: list[str] = ["output", "thought"]
# Words or phrases in observations that are explicitly ignored for progress detection (e.g., "thinking...", "waiting...")
PROGRESS_DETECTOR_IGNORED_PATTERNS: list[str] = ["thinking...", "waiting for user input"]

# Critique Agent Detector
CRITIQUE_AGENT_DETECTOR_ENABLED: bool = False # Set to True to enable LLM-based critique
CRITIQUE_AGENT_MODEL_NAME: str = "gpt-3.5-turbo" # LLM model for critique agent
CRITIQUE_AGENT_TEMPERATURE: float = 0.5       # LLM temperature for critique agent
CRITIQUE_AGENT_PROMPT_TEMPLATE_PATH: str = "monitor/detectors/prompts/critique_prompt.txt" # Path to prompt template
CRITIQUE_AGENT_MAX_TRACE_LENGTH: int = 20     # Max number of recent steps to send to critique agent
CRITIQUE_AGENT_MIN_OBSERVATIONS_BEFORE_CRITIQUE: int = 8 # Min observations before first critique
CRITIQUE_AGENT_PERIODICITY_STEPS: int = 10    # How often to invoke the critique agent (every N steps)

# --- Intervention Settings ---

# Priority order of interventions to attempt when a problem is detected.
# The monitor will try them in this order until one is successful or max attempts are reached.
# Options: 'replan', 'hint', 'human_fallback'
INTERVENTION_PRIORITY_ORDER: list[str] = ["replan", "hint", "human_fallback"]
MAX_INTERVENTIONS_PER_PROBLEM: int = 2 # Max attempts to intervene for a single detected problem instance

# Replan Intervener
REPLAN_INTERVENER_ENABLED: bool = True
REPLAN_STRATEGY: str = "contextual" # Options: 'full_reset', 'contextual'
REPLAN_CONTEXT_INJECTION_TEMPLATE: str = "You seem to be stuck. Please re-evaluate your plan and strategy given the current context: {problem_summary}. Previous goal: {original_goal}"

# Hint Intervener
HINT_INTERVENER_ENABLED: bool = True
HINT_MODEL_NAME: str = "gpt-3.5-turbo" # LLM model for hint generation
HINT_TEMPERATURE: float = 0.6         # LLM temperature for hint generation
HINT_PROMPT_TEMPLATE_PATH: str = "monitor/interventions/prompts/hint_prompt.txt" # Path to prompt template
HINT_MAX_LENGTH: int = 200            # Max length of the generated hint

# Human Fallback Intervener
HUMAN_FALLBACK_ENABLED: bool = True
HUMAN_FALLBACK_EMAIL_RECIPIENTS: list[str] = ["admin@example.com"] # Email addresses for notifications
HUMAN_FALLBACK_WEBHOOK_URL: str = "" # Webhook URL for alerts (e.g., Slack, PagerDuty)
HUMAN_FALLBACK_NOTIFICATION_SUBJECT: str = "AI Agent Stuck - Human Intervention Required"
HUMAN_FALLBACK_MESSAGE_TEMPLATE: str = "Agent {agent_id} is stuck. Problem: {problem_summary}. Trace: {trace_summary}"
HUMAN_FALLBACK_REPORT_PATH: str = "logs/human_fallback_reports/" # Directory to save detailed reports

# --- LLM API Settings (Global defaults if not specified per component) ---
OPENAI_API_KEY: str = "YOUR_OPENAI_API_KEY" # Placeholder: Replace with actual key or env var

# Error Handling Configuration
MAX_TOOL_CALL_RETRIES: int = 3 # Max retries for a single tool call within an agent
RETRY_DELAY_SECONDS: float = 1.0 # Initial delay for retries (exponential backoff might be applied)
ERROR_LOG_PATH: str = "logs/errors.log" # Path for logging critical errors