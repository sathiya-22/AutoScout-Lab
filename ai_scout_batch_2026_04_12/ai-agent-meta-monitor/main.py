```python
import time
import collections
import inspect
import functools
import uuid
import random
from typing import Any, Dict, List, Optional, Tuple, Deque, Callable, Type

# --- Configuration (simplified for prototype) ---
# Allows easy adjustment of detector thresholds, intervention priorities, and agent parameters.
config = {
    "agent_max_steps": 20, # Maximum steps for the agent to run before stopping
    "monitor_check_interval": 1, # Check detectors every X agent steps
    "loop_detector_window_size": 3, # Number of recent steps (e.g., thoughts/tool calls) to check for repeated patterns
    "loop_detector_min_repetitions": 2, # How many times a sequence must repeat to be considered a loop
    "progress_detector_no_change_steps": 5, # How many steps without significant output or internal state change before flagging a stall
    "critique_agent_detector_interval": 7, # Check critique agent every X steps (not strictly enforced here, just a trigger consideration)
    "intervention_priority": { # Priority for interventions; lower number means higher priority
        "replan": 1,
        "hint": 2,
        "human_fallback": 3,
    },
    "agent_initial_goal": "Reach the value 10 by incrementing.",
    "agent_tool_failure_rate": 0.2, # 20% chance the 'increment' tool will fail
    "agent_max_tool_failures_before_loop": 3, # Number of consecutive tool failures to trigger intentional looping in demo agent
}

# --- Shared Data Structures ---
class ProblemReport:
    """
    Carries details about a detected problem from a detector.
    """
    def __init__(self, detector_id: str, problem_type: str, details: Dict[str, Any]):
        self.detector_id = detector_id
        self.problem_type = problem_type
        self.details = details
        self.timestamp = time.time()

    def __repr__(self):
        return f"ProblemReport(detector='{self.detector_id}', type='{self.problem_type}', details={self.details})"

class Observation:
    """
    Represents a single captured event or state change from the agent.
    """
    def __init__(self, step: int, timestamp: float, source: str, data: Dict[str, Any]):
        self.step = step
        self.timestamp = timestamp
        self.source = source # e.g., "thought", "tool_call", "output"
        self.data = data

    def __repr__(self):
        return f"Observation(step={self.step}, source='{self.source}', data={self.data})"

# --- 1. Monitored Agent (`agents/`) ---

# agents/base_agent.py
class BaseAgent:
    """
    Defines an interface for agents, including methods for receiving interventions.
    Agents are designed to expose internal states and accept instrumentation.
    """
    def __init__(self, name: str):
        self.name = name
        self.current_goal: str = ""
        self.is_stuck: bool = False # Agent's internal flag for being stuck
        self.interventions_received: List[Dict[str, Any]] = [] # Record of interventions received

    def run(self, task: str) -> Any:
        """Initializes the agent for a given task."""
        raise NotImplementedError

    def generate_thought(self, prompt: str) -> str:
        """Agent's internal monologue or planning step."""
        raise NotImplementedError

    def tool_call(self, tool_name: str, **kwargs) -> Any:
        """Agent's interaction with external tools."""
        raise NotImplementedError

    def produce_output(self) -> Any:
        """Agent's external output or progress report."""
        raise NotImplementedError

    def replan(self, problem_report: ProblemReport, new_context: Optional[str] = None):
        """Method to trigger agent's re-planning mechanism."""
        print(f"[{self.name}] Agent triggered replan due to: {problem_report.problem_type}")
        self.interventions_received.append({"type": "replan", "report": problem_report, "context": new_context, "timestamp": time.time()})

    def receive_hint(self, hint: str):
        """Method to inject a contextual hint into the agent."""
        print(f"[{self.name}] Agent received hint: {hint}")
        self.interventions_received.append({"type": "hint", "hint": hint, "timestamp": time.time()})

# agents/demo_agent.py
class DemoAgent(BaseAgent):
    """
    A concrete implementation of a simple agent (e.g., ReAct-style)
    that is designed to sometimes exhibit looping or stalling behavior to stress-test the monitor.
    """
    def __init__(self, name: str = "DemoAgent"):
        super().__init__(name)
        self.current_value: int = 0
        self.target_value: int = 0
        self.tool_failure_count: int = 0
        self.max_tool_failures_before_loop: int = config["agent_max_tool_failures_before_loop"]
        self.is_looping_intentionally: bool = False # Flag to simulate agent being stuck
        self.last_thought: str = ""
        self.completed = False

    def run(self, task: str) -> Any:
        print(f"[{self.name}] Starting task: {task}")
        self.current_goal = task
        # Parse target value from task, e.g., "Reach the value 10 by incrementing."
        try:
            self.target_value = int(task.split("value ")[1].split(" ")[0])
        except (IndexError, ValueError):
            self.target_value = 5 # Default if parsing fails or task string is unexpected

        self.current_value = 0
        self.tool_failure_count = 0
        self.is_looping_intentionally = False
        self.completed = False
        self.interventions_received.clear()
        self.is_stuck = False
        return self.completed # Indicates initial status

    def generate_thought(self, prompt: str) -> str:
        if self.completed:
            self.last_thought = "Task completed."
            return self.last_thought

        if self.is_looping_intentionally:
            self.last_thought = f"I am stuck. I need to increment {self.current_value} but it's not working. Retrying the same action because I can't think of anything else."
            self.is_stuck = True
            return self.last_thought
        
        if self.current_value >= self.target_value:
            self.last_thought = "The current value has reached or exceeded the target. I should produce the final output."
        else:
            self.last_thought = f"The current value is {self.current_value}, target is {self.target_value}. I need to increment."
        
        self.is_stuck = False
        return self.last_thought

    def tool_call(self, tool_name: str, **kwargs) -> Any:
        if self.completed:
            print(f"[{self.name}] Skipping tool call '{tool_name}' as task is completed.")
            return {"status": "skipped", "message": "Task completed."}

        if tool_name == "increment":
            amount = kwargs.get("amount", 1)
            # Simulate failure to trigger loop/stall detection
            if random.random() < config["agent_tool_failure_rate"] and not self.is_looping_intentionally:
                self.tool_failure_count += 1
                if self.tool_failure_count >= self.max_tool_failures_before_loop:
                    print(f"[{self.name}] Tool '{tool_name}' failed too many times, entering intentional loop state.")
                    self.is_looping_intentionally = True
                    self.is_stuck = True
                print(f"[{self.name}] Tool '{tool_name}' failed. Current failures: {self.tool_failure_count}")
                return {"status": "failed", "message": "Simulated transient error."}
            else:
                if self.is_looping_intentionally:
                    # Once in intentional loop, if intervention happens and tool succeeds, reset
                    print(f"[{self.name}] Tool '{tool_name}' succeeded in loop state, resetting loop flag.")
                    self.is_looping_intentionally = False
                    self.is_stuck = False
                    self.tool_failure_count = 0 # Reset on success
                self.current_value += amount
                print(f"[{self.name}] Incremented value by {amount}. Current value: {self.current_value}")
                return {"status": "success", "new_value": self.current_value}
        else:
            print(f"[{self.name}] Unknown tool: {tool_name}")
            return {"status": "error", "message": f"Unknown tool '{tool_name}'"}

    def produce_output(self) -> Any:
        if self.completed:
            return f"Final output: Task already completed, value is {self.current_value}"

        if self.current_value >= self.target_value:
            output = f"Task completed successfully. Final value is {self.current_value} (target was {self.target_value})."
            self.completed = True
        else:
            output = f"Current progress: value is {self.current_value} (target {self.target_value}). Still working."
        
        print(f"[{self.name}] Producing output: {output}")
        return output

    def replan(self, problem_report: ProblemReport, new_context: Optional[str] = None):
        super().replan(problem_report, new_context)
        print(f"[{self.name}] Agent is replanning. Resetting tool failure count and loop flag to adapt.")
        self.is_looping_intentionally = False
        self.is_stuck = False
        self.tool_failure_count = 0
        # Agent would internally use new_context to modify its plan or strategy
        if new_context:
            print(f"[{self.name}] Incorporating new context: {new_context}")

    def receive_hint(self, hint: str):
        super().receive_hint(hint)
        print(f"[{self.name}] Agent processing hint. Resetting tool failure count and loop flag to adapt.")
        self.is_looping_intentionally = False
        self.is_stuck = False
        self.tool_failure_count = 0
        # Agent would use the hint to adjust its next thought or action
        if "consider increasing the increment amount" in hint:
            print(f"[{self.name}] Hint suggests larger increment. Will consider in next step.")

# --- 4. State Manager (`monitor/state_manager.py`) ---
class StateManager:
    """
    Acts as the monitor's memory, maintaining a historical trace of agent actions,
    thoughts, tool calls, and outputs.
    """
    def __init__(self, max_history_size: int = 100):
        self.history: Deque[Observation] = collections.deque(maxlen=max_history_size)
        self.current_step_count = 0 # Tracks the logical steps of the agent
        self._output_history: Deque[Any] = collections.deque(maxlen=max_history_size) # Stores only output values for progress detection

    def add_observation(self, source: str, data: Dict[str, Any]):
        """Adds a new observation to the history."""
        observation = Observation(self.current_step_count, time.time(), source, data)
        self.history.append(observation)
        if source == "output":
            self._output_history.append(data.get("output_value", None))
        # print(f"[StateManager] Added observation: {observation}") # Verbose logging

    def get_history(self) -> List[Observation]:
        """Returns the full historical trace."""
        return list(self.history)

    def get_output_history(self) -> List[Any]:
        """Returns only the history of output values."""
        return list(self._output_history)

    def increment_step(self):
        """Increments the current step counter."""
        self.current_step_count += 1

    def get_current_step(self) -> int:
        """Returns the current step count."""
        return self.current_step_count

# --- 3. Instrumentation (`monitor/instrumentation.py`) ---
class Instrumentation:
    """
    Provides decorators or wrapper functions to capture agent internal states and
    external feedback, forwarding them to the StateManager.
    """
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        # Store original methods to avoid double-instrumentation or for potential uninstrumentation
        self.instrumented_methods: Dict[Tuple[Any, str], Callable] = {}

    def instrument_method(self, agent_instance: Any, method_name: str, source_type: str):
        """
        Wraps a specified method of an agent instance to capture its execution details.
        """
        original_method = getattr(agent_instance, method_name)
        if (agent_instance, method_name) in self.instrumented_methods:
            print(f"Warning: Method '{method_name}' on {agent_instance.name} already instrumented. Skipping.")
            return

        @functools.wraps(original_method)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = original_method(*args, **kwargs) # Execute original method
            end_time = time.time()

            # Construct observation data
            observation_data = {
                "method_name": method_name,
                "args": args,
                "kwargs": kwargs,
                "result": result,
                "duration_ms": (end_time - start_time) * 1000,
                "agent_state": { # Snapshot of key agent internal states
                    "current_value": getattr(agent_instance, 'current_value', None),
                    "target_value": getattr(agent_instance, 'target_value', None),
                    "is_looping_intentionally": getattr(agent_instance, 'is_looping_intentionally', None),
                    "is_stuck": getattr(agent_instance, 'is_stuck', None),
                    "completed": getattr(agent_instance, 'completed', None),
                    "tool_failure_count": getattr(agent_instance, 'tool_failure_count', None),
                    "last_thought": getattr(agent_instance, 'last_thought', None),
                }
            }
            # Add type-specific details to the observation
            if source_type == "tool_call":
                observation_data["tool_name"] = args[0] if args else "unknown"
                observation_data["tool_status"] = result.get("status")
            elif source_type == "thought":
                observation_data["thought_content"] = result
            elif source_type == "output":
                observation_data["output_value"] = result
            
            self.state_manager.add_observation(source_type, observation_data)
            return result
        
        setattr(agent_instance, method_name, wrapper)
        self.instrumented_methods[(agent_instance, method_name)] = original_method # Store original method

    def instrument_agent(self, agent_instance: BaseAgent):
        """Applies instrumentation to the key methods of an agent instance."""
        print(f"[Instrumentation] Instrumenting agent: {agent_instance.name}")
        self.instrument_method(agent_instance, "generate_thought", "thought")
        self.instrument_method(agent_instance, "tool_call", "tool_call")
        self.instrument_method(agent_instance, "produce_output", "output")


# --- 5. Detectors (`monitor/detectors/`) ---

# monitor/detectors/base_detector.py
class BaseDetector:
    """Defines the interface for all detectors."""
    def __init__(self, detector_id: str):
        self.detector_id = detector_id

    def detect(self, state_history: List[Observation]) -> Optional[ProblemReport]:
        """
        Analyzes the agent's state history to identify problems.
        Returns a ProblemReport if an issue is found, otherwise None.
        """
        raise NotImplementedError

# monitor/detectors/loop_detector.py
class LoopDetector(BaseDetector):
    """
    Implements heuristics to detect repeated sequences of actions, thoughts,
    or observations within a defined window.
    """
    def __init__(self, detector_id: str = "LoopDetector", window_size: int = 3, min_repetitions: int = 2):
        super().__init__(detector_id)
        self.window_size = window_size
        self.min_repetitions = min_repetitions

    def detect(self, state_history: List[Observation]) -> Optional[ProblemReport]:
        # Need enough history to compare at least two windows
        if len(state_history) < self.window_size * self.min_repetitions:
            return None 

        # Extract relevant sequence elements for pattern matching (e.g., thoughts or tool calls)
        # Combine thought content and tool call status to form a comparable sequence
        sequence_elements: List[str] = []
        # Look at recent history, at least (window_size * min_repetitions) steps
        recent_history = state_history[-(self.window_size * self.min_repetitions * 2):] # Look at a larger chunk to be safe

        for obs in recent_history:
            if obs.source == "thought":
                # Use a hash or truncated string for thought content to handle long thoughts
                thought_summary = obs.data.get('thought_content', '')[:50] # Truncate for comparison
                sequence_elements.append(f"T:{thought_summary}")
            elif obs.source == "tool_call":
                status = obs.data.get('tool_status', 'N/A')
                tool_name = obs.data.get('tool_name', 'N/A')
                sequence_elements.append(f"C:{tool_name}({status})")

        if len(sequence_elements) < self.window_size * self.min_repetitions:
             return None # Still not enough relevant history

        # Check for repeating patterns by iterating possible pattern lengths
        for i in range(self.window_size, len(sequence_elements) // self.min_repetitions + 1):
            if i == 0: continue # Avoid division by zero or empty pattern
            pattern = tuple(sequence_elements[-i:]) # The most recent sequence of length 'i'
            
            # Check if this pattern repeats immediately before
            if len(pattern) > 0 and len(sequence_elements) >= 2 * len(pattern):
                previous_pattern = tuple(sequence_elements[-2*len(pattern):-len(pattern)])
                if pattern == previous_pattern:
                    return ProblemReport(
                        detector_id=self.detector_id,
                        problem_type="Repetitive Loop Detected",
                        details={
                            "repeated_pattern": list(pattern),
                            "window_size": len(pattern),
                            "repetitions": 2, # Detected at least two immediate repetitions
                            "full_sequence_checked": sequence_elements[-2*len(pattern):]
                        }
                    )
        return None

# monitor/detectors/progress_detector.py
class ProgressDetector(BaseDetector):
    """
    Monitors for lack of change in the agent's internal state or output
    over a threshold duration or number of steps.
    """
    def __init__(self, detector_id: str = "ProgressDetector", no_change_steps: int = 5):
        super().__init__(detector_id)
        self.no_change_steps = no_change_steps

    def detect(self, state_history: List[Observation]) -> Optional[ProblemReport]:
        if len(state_history) < self.no_change_steps:
            return None

        # 1. Check for lack of change in outputs
        recent_outputs = [obs.data.get("output_value") for obs in state_history if obs.source == "output"]
        if len(recent_outputs) >= self.no_change_steps:
            if all(output == recent_outputs[-1] for output in recent_outputs[-self.no_change_steps:]):
                return ProblemReport(
                    detector_id=self.detector_id,
                    problem_type="Output Stall Detected",
                    details={
                        "last_output_value": recent_outputs[-1],
                        "steps_without_change": self.no_change_steps
                    }
                )
        
        # 2. Check for lack of change in agent's internal state (current_value)
        # This relies on the agent state being captured in observations
        recent_current_values = []
        for obs in state_history[-self.no_change_steps:]:
            agent_state = obs.data.get("agent_state", {})
            if "current_value" in agent_state:
                recent_current_values.append(agent_state["current_value"])
        
        if len(recent_current_values) >= self.no_change_steps:
            if all(val == recent_current_values[-1] for val in recent_current_values[-self.no_change_steps:]):
                # Ensure that the agent hasn't completed the task, as a final stable value is not a stall
                if state_history[-1].data.get("agent_state", {}).get("completed") is False:
                    return ProblemReport(
                        detector_id=self.detector_id,
                        problem_type="Internal State Stall Detected (e.g., current_value not changing)",
                        details={
                            "last_known_value": recent_current_values[-1],
                            "steps_without_change": self.no_change_steps
                        }
                    )
        
        return None

# monitor/detectors/critique_agent_detector.py
class CritiqueAgentDetector(BaseDetector):
    """
    Leverages a separate, smaller LLM (simulated here) to analyze a segment of the
    agent's trace and identify higher-level reasoning flaws, loops, or stalls.
    """
    def __init__(self, detector_id: str = "CritiqueAgentDetector"):
        super().__init__(detector_id)
        # In a real scenario, this would initialize an LLM client, e.g., self.critique_llm = LLMClient(...)

    def detect(self, state_history: List[Observation]) -> Optional[ProblemReport]:
        # For prototype, simulate critique agent analysis based on agent's reported state
        if not state_history:
            return None
        
        last_agent_state = state_history[-1].data.get("agent_state", {})
        
        # Scenario 1: Agent reports being intentionally stuck
        if last_agent_state.get("is_looping_intentionally") is True:
            return ProblemReport(
                detector_id=self.detector_id,
                problem_type="Critique Agent Confirmed Loop/Stall (Self-Reported)",
                details={
                    "reasoning": "Critique agent analyzed the agent's internal state and found it reports being intentionally stuck or looping.",
                    "agent_reported_state": last_agent_state
                }
            )
        
        # Scenario 2: High number of tool failures without completion
        if last_agent_state.get("tool_failure_count", 0) > (config["agent_max_tool_failures_before_loop"] - 1) and \
           last_agent_state.get("completed") is False:
            # Simulate LLM reasoning that repeated failures likely mean a stall
            return ProblemReport(
                detector_id=self.detector_id,
                problem_type="Critique Agent Suspects Stall from Repeated Tool Failures",
                details={
                    "reasoning": "Critique agent observed repeated tool failures without task completion or change in strategy.",
                    "tool_failure_count": last_agent_state.get("tool_failure_count")
                }
            )
        
        # Scenario 3: Agent is stuck in the same thought for too long (even if not self-reported loop)
        recent_thoughts = [obs.data.get('thought_content') for obs in state_history if obs.source == 'thought']
        if len(recent_thoughts) >= config["progress_detector_no_change_steps"]:
            if all(t == recent_thoughts[-1] for t in recent_thoughts[-config["progress_detector_no_change_steps"]:]):
                if last_agent_state.get("completed") is False:
                    return ProblemReport(
                        detector_id=self.detector_id,
                        problem_type="Critique Agent Suspects Mental Loop",
                        details={
                            "reasoning": "Critique agent observes the agent repeating the same thought pattern without progress.",
                            "last_thought": recent_thoughts[-1],
                            "repeated_steps": config["progress_detector_no_change_steps"]
                        }
                    )

        return None


# --- 6. Interventions (`monitor/interventions/`) ---

# monitor/interventions/base_intervener.py
class BaseIntervener:
    """Defines the interface for all interveners."""
    def __init__(self, intervener_id: str, priority: int = 0):
        self.intervener_id = intervener_id
        self.priority = priority # Lower number means higher priority

    def intervene(self, agent: BaseAgent, problem_report: ProblemReport) -> bool:
        """
        Applies an intervention to the agent.
        Returns True if intervention was successfully applied, False otherwise.
        """
        raise NotImplementedError

# monitor/interventions/replan_intervener.py
class ReplanIntervener(BaseIntervener):
    """Triggers the monitored agent's re-planning mechanism."""
    def __init__(self, intervener_id: str = "ReplanIntervener", priority: int = 1):
        super().__init__(intervener_id, priority)

    def intervene(self, agent: BaseAgent, problem_report: ProblemReport) -> bool:
        print(f"[Intervention] {self.intervener_id} activating for {agent.name} due to: {problem_report.problem_type}")
        new_context = f"The monitor detected a problem: '{problem_report.problem_type}'. Details: {problem_report.details}. Please re-evaluate your strategy, identify the root cause of the loop/stall, and generate a new plan to achieve your goal '{agent.current_goal}'."
        agent.replan(problem_report, new_context)
        return True

# monitor/interventions/hint_intervener.py
class HintIntervener(BaseIntervener):
    """Injects a contextual hint or instruction into the agent's prompt or internal monologue."""
    def __init__(self, intervener_id: str = "HintIntervener", priority: int = 2):
        super().__init__(intervener_id, priority)

    def intervene(self, agent: BaseAgent, problem_report: ProblemReport) -> bool:
        print(f"[Intervention] {self.intervener_id} activating for {agent.name} due to: {problem_report.problem_type}")
        hint_message = f"Monitor suggests: It seems you're stuck in a {problem_report.problem_type}. Consider a different tool or re-evaluate your approach to '{agent.current_goal}'."
        if "Output Stall Detected" in problem_report.problem_type:
            hint_message += " Perhaps your criteria for task completion needs adjustment, or there's an issue with the environment feedback."
        elif "Repetitive Loop Detected" in problem_report.problem_type:
            hint_message += " Try breaking the pattern. Can you explore an alternative path or re-interpret the problem statement?"
        elif "Critique Agent" in problem_report.problem_type:
            hint_message += " The critique agent has a strong suspicion you are off track. Take this warning seriously and re-evaluate thoroughly."

        agent.receive_hint(hint_message)
        return True

# monitor/interventions/human_fallback.py
class HumanFallback(BaseIntervener):
    """
    Notifies a human operator or logs a detailed report for manual intervention
    when automated interventions are insufficient or deemed high-risk.
    """
    def __init__(self, intervener_id: str = "HumanFallback", priority: int = 3):
        super().__init__(intervener_id, priority)
        self.state_manager_ref: Optional[StateManager] = None # Will be set by MonitorCore

    def set_state_manager(self, state_manager: StateManager):
        self.state_manager_ref = state_manager

    def intervene(self, agent: BaseAgent, problem_report: ProblemReport) -> bool:
        print(f"\n{'='*70}\n[!!! HUMAN INTERVENTION REQUIRED !!!]")
        print(f"[{self.intervener_id}] Notifying human operator about {problem_report.problem_type} for agent '{agent.name}'.")
        print(f"Problem: {problem_report.problem_type}")
        print(f"Details: {problem_report.details}")
        print(f"Agent's last known goal: {agent.current_goal}")
        print(f"Agent's last thought: {agent.last_thought}")
        print(f"Agent's internal state (current_value={agent.current_value}, target_value={agent.target_value}, completed={agent.completed}, stuck={agent.is_stuck}, tool_failures={agent.tool_failure_count})")
        if self.state_manager_ref:
            history_snippet = self.state_manager_ref.get_history()
            print(f"Monitor History Snippet (last 5 observations):\n")
            for obs in list(history_snippet)[-5:]:
                print(f"  [{obs.step}] {obs.source}: {obs.data}")
        else:
            print("Monitor history not available for human fallback.")
        print(f"{'='*70}\n")
        return True # Signify that a fallback was initiated.

# --- 2. Monitor Core (`monitor/core.py`) ---
class MonitorCore:
    """
    The central orchestrator. It initializes and manages the StateManager,
    Instrumentation, various Detectors, and Interventions.
    """
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.detectors: List[BaseDetector] = []
        self.interveners: List[BaseIntervener] = [] # Sorted by priority
        self.last_intervention_step: int = -1 # To prevent repeated interventions in the same step
        self.total_interventions_applied = 0

    def add_detector(self, detector: BaseDetector):
        """Registers a detector with the monitor."""
        self.detectors.append(detector)
        print(f"[MonitorCore] Registered detector: {detector.detector_id}")

    def add_intervener(self, intervener: BaseIntervener):
        """Registers an intervener and keeps the list sorted by priority."""
        self.interveners.append(intervener)
        self.interveners.sort(key=lambda i: i.priority) # Keep interveners sorted by priority
        print(f"[MonitorCore] Registered intervener: {intervener.intervener_id} (Priority: {intervener.priority})")
        # If it's the HumanFallback, provide a reference to StateManager
        if isinstance(intervener, HumanFallback):
            intervener.set_state_manager(self.state_manager)

    def check_for_problems_and_intervene(self, agent: BaseAgent) -> bool:
        """
        Queries registered detectors for issues and triggers appropriate interventions.
        Returns True if an intervention was applied, False otherwise.
        """
        current_step = self.state_manager.get_current_step()
        if current_step == self.last_intervention_step:
            # Prevent immediate re-intervention within the same agent step
            return False

        problem_reports: List[ProblemReport] = []
        for detector in self.detectors:
            report = detector.detect(self.state_manager.get_history())
            if report:
                problem_reports.append(report)
                print(f"[MonitorCore] Detected problem: {report.problem_type} by {detector.detector_id}")

        if problem_reports:
            # Prioritize a ProblemReport for intervention based on detector type
            # For example, CritiqueAgentDetector might signal a more critical issue
            primary_report = problem_reports[0]
            for report in problem_reports:
                if report.detector_id == "CritiqueAgentDetector":
                    primary_report = report # Prioritize critique agent reports
                    break
            
            # Select the highest priority intervener that is available
            selected_intervener: Optional[BaseIntervener] = None
            if self.interveners:
                selected_intervener = self.interveners[0] # Highest priority is at index 0

            if selected_intervener:
                print(f"[MonitorCore] Attempting intervention '{selected_intervener.intervener_id}' (Priority: {selected_intervener.priority})")
                intervention_success = selected_intervener.intervene(agent, primary_report)
                if intervention_success:
                    self.total_interventions_applied += 1
                    self.last_intervention_step = current_step
                    return True
            else:
                print("[MonitorCore] No interveners registered to handle detected problems.")

        return False


# --- Main Demonstration Logic (`main.py`) ---
def main():
    print("--- Starting AI Meta-Monitoring Prototype ---")

    # 1. Instantiate Monitored Agent
    demo_agent = DemoAgent(name="ComplexTaskAgent")

    # 2. Instantiate Monitor Core Components
    state_manager = StateManager(max_history_size=config["agent_max_steps"] * 2) # Enough history for analysis
    instrumentation = Instrumentation(state_manager)
    monitor_core = MonitorCore(state_manager)

    # 3. Register Detectors with the Monitor Core
    monitor_core.add_detector(LoopDetector(
        window_size=config["loop_detector_window_size"],
        min_repetitions=config["loop_detector_min_repetitions"]
    ))
    monitor_core.add_detector(ProgressDetector(
        no_change_steps=config["progress_detector_no_change_steps"]
    ))
    monitor_core.add_detector(CritiqueAgentDetector())

    # 4. Register Interveners with the Monitor Core (sorted by priority from config)
    sorted_intervener_ids = sorted(config["intervention_priority"], key=config["intervention_priority"].get)
    for intervener_id in sorted_intervener_ids:
        priority = config["intervention_priority"][intervener_id]
        if intervener_id == "replan":
            monitor_core.add_intervener(ReplanIntervener(priority=priority))
        elif intervener_id == "hint":
            monitor_core.add_intervener(HintIntervener(priority=priority))
        elif intervener_id == "human_fallback":
            monitor_core.add_intervener(HumanFallback(priority=priority))
        else:
            print(f"Warning: Unknown intervener ID '{intervener_id}' in config. Skipping.")

    # 5. Instrument the Agent: Wrap its key methods to send observations to StateManager
    instrumentation.instrument_agent(demo_agent)

    # 6. Define and Start the Sample Task
    initial_task = config["agent_initial_goal"]
    demo_agent.run(initial_task) # Initializes agent's internal state for the task

    max_steps = config["agent_max_steps"]
    steps_taken = 0
    task_completed = False

    print(f"\n--- Running Agent '{demo_agent.name}' for max {max_steps} steps ---")

    while steps_taken < max_steps and not demo_agent.completed:
        steps_taken += 1
        state_manager.increment_step() # Increment monitor's step counter for observations
        print(f"\n--- STEP {steps_taken} ---")

        # Agent performs its sequential actions for this step
        # 1. Generate a thought
        thought_prompt = f"Current goal: {demo_agent.current_goal}. Current value: {demo_agent.current_value}. Target: {demo_agent.target_value}. What's next?"
        demo_agent.generate_thought(thought_prompt) # This call is instrumented

        # 2. Potentially call a tool based on the thought
        # Simple logic: if thought suggests incrementing, call the tool
        if "increment" in demo_agent.last_thought.lower():
            demo_agent.tool_call("increment", amount=1) # This call is instrumented
        
        # 3. Produce output (can be a progress report or final result)
        demo_agent.produce_output() # This call is instrumented

        # Check if the agent has completed the task after its actions
        if demo_agent.completed:
            task_completed = True
            break # Exit simulation loop if task is done
        
        # Monitor checks for problems and applies interventions periodically
        if steps_taken % config["monitor_check_interval"] == 0:
            intervention_applied = monitor_core.check_for_problems_and_intervene(demo_agent)
            if intervention_applied:
                # If a human fallback intervention was triggered, we might want to halt the simulation
                if isinstance(monitor_core.interveners[0], HumanFallback): # HumanFallback is likely highest priority if configured this way
                    # Check if the last applied intervention was indeed a human fallback
                    # This is a bit simplistic, but for a prototype, it works.
                    # A more robust check would involve the ProblemReport's details.
                    if monitor_core.total_interventions_applied > 0 and monitor_core.interveners[0].intervener_id == "HumanFallback":
                        print("[Main] Human Fallback triggered as the highest priority intervention. Halting simulation.")
                        break

        # A small delay to simulate processing time and make the output readable
        time.sleep(0.1) 
    
    print("\n--- Simulation Complete ---")
    print(f"Total steps taken: {steps_taken}")
    print(f"Task completed: {task_completed}")
    print(f"Final Agent Value: {demo_agent.current_value}")
    print(f"Target Value: {demo_agent.target_value}")
    print(f"Total interventions applied: {monitor_core.total_interventions_applied}")

    if not task_completed:
        print("\nAgent did not complete the task within the maximum steps.")
        print(f"Agent's last thought: {demo_agent.last_thought}")
        print(f"Agent's current goal: {demo_agent.current_goal}")
        print(f"Agent's final state: current_value={demo_agent.current_value}, stuck_flag={demo_agent.is_stuck}, tool_failure_count={demo_agent.tool_failure_count}")
    else:
        print("\nAgent successfully completed the task!")

    # Optional: print interventions received by the agent
    if demo_agent.interventions_received:
        print("\n--- Agent's Intervention History ---")
        for intervention in demo_agent.interventions_received:
            print(f"- Type: {intervention['type']}, Timestamp: {time.strftime('%H:%M:%S', time.localtime(intervention['timestamp']))}")
            if 'hint' in intervention: print(f"  Hint: {intervention['hint'][:70]}...")
            if 'report' in intervention: print(f"  Report type: {intervention['report'].problem_type}")
    else:
        print("\nNo interventions were applied to the agent.")


if __name__ == "__main__":
    main()
```