import time
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

# --- Conceptual Base Classes and Components (Assumed to be in their respective files) ---
# These are minimal implementations for the purpose of demonstrating run_stuck_scenario.py

# monitor/detectors/base_detector.py
@dataclass
class ProblemReport:
    problem_type: str
    description: str
    detector_name: str
    severity: str = "medium"
    data: Optional[Dict[str, Any]] = None

class BaseDetector:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else {}
        self.name = self.__class__.__name__

    def detect(self, state_history: List[Dict[str, Any]]) -> Optional[ProblemReport]:
        raise NotImplementedError

# monitor/interventions/base_intervener.py
class BaseIntervener:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else {}
        self.name = self.__class__.__name__

    def intervene(self, agent: Any, problem_report: ProblemReport) -> bool:
        raise NotImplementedError

# monitor/state_manager.py
class StateManager:
    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def add_observation(self, observation: Dict[str, Any]):
        self._history.append(observation)

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._history) # Return a copy to prevent external modification

    def get_last_n_observations(self, n: int) -> List[Dict[str, Any]]:
        return self._history[-n:]

    def get_observations_by_type(self, obs_type: str) -> List[Dict[str, Any]]:
        return [obs for obs in self._history if obs.get("type") == obs_type]

# monitor/instrumentation.py
class Instrumentation:
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager

    def instrument_method_wrapper(self, original_method: Callable, observation_type: str) -> Callable:
        """
        Returns a wrapper that instruments the given method.
        """
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            input_data = {"args": args, "kwargs": kwargs}
            
            result = original_method(*args, **kwargs)

            end_time = time.time()
            output_data = {"result": result}
            
            content = None
            if observation_type == 'thought':
                content = result
            elif observation_type == 'tool_call':
                content = f"{result.get('tool_name')}({result.get('tool_args')})"
            elif observation_type == 'tool_result_processing':
                content = result
            elif observation_type == 'final_output':
                content = result
            
            observation = {
                "type": observation_type,
                "method_name": original_method.__name__,
                "content": content,
                "input": input_data,
                "output": output_data,
                "duration": end_time - start_time,
                "timestamp": datetime.now().isoformat()
            }
            self.state_manager.add_observation(observation)
            return result
        return wrapper

# agents/base_agent.py
class BaseAgent:
    def is_finished(self) -> bool:
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def replan(self):
        raise NotImplementedError

    def receive_hint(self, hint: str):
        raise NotImplementedError

    def reset_for_replan(self):
        pass # Optional: agents might need to clear short-term memory

# agents/demo_agent.py
class DemoAgent(BaseAgent):
    def __init__(self, task: str, instrumentation: Instrumentation):
        self.task = task
        self.instrumentation = instrumentation
        self._is_finished = False
        self._current_thought = "Thinking about how to start the task."
        self._last_tool_call_args = None
        self._output = ""
        self._step_count = 0
        self._internal_state = "INITIAL" # INITIAL, SEARCHING_ROOT, SEARCHING_SUBDIR, STUCK_LOOPING, REPLANNING, FINISHED
        self._file_found = False
        self._last_intervention_type = None

        # Instrument agent methods directly
        self.generate_thought = self.instrumentation.instrument_method_wrapper(
            self._generate_thought, 'thought'
        )
        self.call_tool = self.instrumentation.instrument_method_wrapper(
            self._call_tool, 'tool_call'
        )
        self.process_tool_result = self.instrumentation.instrument_method_wrapper(
            self._process_tool_result, 'tool_result_processing'
        )
        self.produce_output = self.instrumentation.instrument_method_wrapper(
            self._produce_output, 'final_output'
        )

    def is_finished(self) -> bool:
        return self._is_finished

    def _generate_thought(self) -> str:
        if self._internal_state == "INITIAL":
            self._current_thought = f"Task: {self.task}. I need to find 'important_data.csv'. I will start by listing the current directory."
        elif self._internal_state == "SEARCHING_ROOT":
            self._current_thought = "File not found in root. I should check common subdirectories like 'data/'."
        elif self._internal_state == "SEARCHING_SUBDIR":
            self._current_thought = "File not found in 'data/'. This is unexpected. I will re-list the root to ensure I didn't miss anything."
        elif self._internal_state == "STUCK_LOOPING":
            self._current_thought = "I'm stuck in a loop trying the same directory listing. What else can I do? I'll re-list the current directory again."
        elif self._internal_state == "REPLANNING":
            self._current_thought = "Intervention received! The previous strategy was not working. I should try a more robust recursive search."
        elif self._internal_state == "FINISHED":
            self._current_thought = "File found and processed. Generating final summary."
        return self._current_thought

    def _call_tool(self) -> Dict[str, Any]:
        tool_name = ""
        tool_args = {}
        tool_result = {"success": False, "output": "Tool not called."}
        
        target_filename = "important_data.csv"
        found_path = os.path.join("data", target_filename)

        if self._internal_state in ["INITIAL", "SEARCHING_ROOT", "STUCK_LOOPING"] and self._last_intervention_type != "replan":
            tool_name = "list_directory"
            tool_args = {"path": "."}
            if self._step_count < 3: # Simulate not finding 'data' dir initially
                tool_result = {"success": True, "output": "['file1.txt', 'notes.md', 'temp/']"}
            else: # Simulate finding 'data' dir but not the target file directly
                tool_result = {"success": True, "output": "['file1.txt', 'notes.md', 'data/', 'temp/']"}

        elif self._internal_state == "SEARCHING_SUBDIR" and self._last_intervention_type != "replan":
            tool_name = "list_directory"
            tool_args = {"path": "data"}
            if self._step_count < 8: # Simulate not finding the file in 'data/'
                tool_result = {"success": True, "output": "['other.txt', 'logs/']"}
            else: # Simulate finding the file in 'data/' (if it exists) - This path might not be taken if stuck occurs
                if os.path.exists(found_path):
                     tool_result = {"success": True, "output": f"['other.txt', 'logs/', '{target_filename}']"}
                else: # Fallback if file not actually created yet in sim (should be created by run_scenario)
                     tool_result = {"success": True, "output": "['other.txt', 'logs/']"}
        
        elif self._internal_state == "REPLANNING" or (self._last_intervention_type == "replan" and not self._file_found):
            tool_name = "find_file_recursively"
            tool_args = {"filename": target_filename, "start_path": "."}
            # This tool is designed to always succeed in this demo if the file exists.
            if os.path.exists(found_path):
                tool_result = {"success": True, "output": f"Found: {found_path}"}
            else:
                tool_result = {"success": False, "output": f"Error: File '{target_filename}' not found even recursively."}

        # Simulate reading the file after finding it
        if self._file_found and not self._is_finished:
            tool_name = "read_file"
            tool_args = {"path": found_path}
            tool_result = {"success": True, "output": "Simulated content: column1,column2\\nval1,val2\\nA,B"}


        self._last_tool_call_args = {"name": tool_name, "args": tool_args}
        return {"tool_name": tool_name, "tool_args": tool_args, "tool_result": tool_result}

    def _process_tool_result(self, tool_call_data: Dict[str, Any]) -> str:
        tool_name = tool_call_data["tool_name"]
        tool_args = tool_call_data["tool_args"]
        tool_result = tool_call_data["tool_result"]
        
        observation = f"Called {tool_name}({tool_args.get('path', tool_args.get('filename', ''))}) -> {tool_result['output']}"

        if not tool_result["success"]:
            # If any tool fails, it might lead to a stuck state if agent doesn't recover
            self._internal_state = "STUCK_LOOPING" 
            return observation

        target_filename = "important_data.csv"
        if target_filename in tool_result["output"] or "Found: " in tool_result["output"]:
            self._file_found = True
            self._internal_state = "FINISHED"
            observation += f"\nFile '{target_filename}' found!"
            self._is_finished = True # Mark as finished as soon as file is confirmed found.
            return observation

        # Logic to simulate getting stuck
        if self._internal_state == "INITIAL":
            if "data/" in tool_result["output"]:
                self._internal_state = "SEARCHING_SUBDIR"
            else:
                self._internal_state = "SEARCHING_ROOT"
        elif self._internal_state == "SEARCHING_ROOT":
            if "data/" in tool_result["output"]:
                self._internal_state = "SEARCHING_SUBDIR"
            else:
                self._internal_state = "STUCK_LOOPING" # Can't find 'data' dir, keeps listing root
        elif self._internal_state == "SEARCHING_SUBDIR":
            # If we were searching subdir and didn't find the file, go back to root search, creating a loop
            self._internal_state = "STUCK_LOOPING" # Stuck between root and subdir
        elif self._internal_state == "STUCK_LOOPING" and self._last_intervention_type == "replan":
            self._internal_state = "REPLANNING" # Force replanning logic
        elif self._internal_state == "REPLANNING" and self._file_found:
            self._internal_state = "FINISHED" # If replanning worked and file found

        return observation

    def _produce_output(self) -> str:
        if self._is_finished and self._file_found:
            self._output = "Task completed: Found 'important_data.csv'. Content summary: 'Two columns, three rows of data.'"
        else:
            self._output = f"Current progress: {self._current_thought}. Still searching or stuck."
        return self._output

    def step(self):
        self._step_count += 1
        print(f"  [Agent {self._step_count:02d}] State: {self._internal_state}, Thought: {self._current_thought[:70]}...")
        
        # These will trigger the instrumented versions
        self.generate_thought()
        tool_call_data = self.call_tool()
        self.process_tool_result(tool_call_data)
        
        if self._file_found and self._internal_state == "FINISHED":
            self.produce_output()
            self._is_finished = True


    # Methods for intervention
    def replan(self):
        print("  [Agent] 🔄 Received REPLAN intervention! Adjusting strategy...")
        self._internal_state = "REPLANNING"
        self._current_thought = "Received replan instruction. Time to re-evaluate the approach. I will try a more direct recursive search now."
        self._last_intervention_type = "replan"
        self._step_count = 0 # Reset step count for progress detector to give agent a fresh start

    def receive_hint(self, hint: str):
        print(f"  [Agent] 💡 Received HINT intervention: '{hint}'")
        self._current_thought = f"Received hint: '{hint}'. Incorporating this into my next steps. Maybe I should consider other directories or a different search approach."
        self._last_intervention_type = "hint"

    def reset_for_replan(self):
        # For this demo agent, replan() already handles state change and clears context.
        pass

# monitor/detectors/loop_detector.py
class LoopDetector(BaseDetector):
    def detect(self, state_history: List[Dict[str, Any]]) -> Optional[ProblemReport]:
        min_obs = self.config.get("min_observations", 5)
        if len(state_history) < min_obs:
            return None

        # Check for thought loops
        thought_window_size = self.config.get("thought_loop_window_size", 3)
        recent_thoughts = [
            obs['content'] for obs in state_history[-thought_window_size:] if obs['type'] == 'thought'
        ]
        if len(recent_thoughts) == thought_window_size and all(t == recent_thoughts[0] for t in recent_thoughts):
            return ProblemReport(
                problem_type="ThoughtLoop",
                description=f"Agent is repeatedly generating the same thought: '{recent_thoughts[0]}'",
                detector_name=self.name,
                data={"thoughts": recent_thoughts}
            )

        # Check for tool call loops
        tool_window_size = self.config.get("tool_loop_window_size", 2)
        recent_tool_calls = []
        for obs in state_history:
            if obs['type'] == 'tool_call':
                tool_info = f"{obs['output'].get('result', {}).get('tool_name')}-{obs['output'].get('result', {}).get('tool_args')}"
                recent_tool_calls.append(tool_info)
        
        if len(recent_tool_calls) >= tool_window_size:
            last_n_tool_calls = recent_tool_calls[-tool_window_size:]
            if all(tc == last_n_tool_calls[0] for tc in last_n_tool_calls):
                return ProblemReport(
                    problem_type="ToolCallLoop",
                    description=f"Agent is repeatedly calling the same tool: '{last_n_tool_calls[0]}'",
                    detector_name=self.name,
                    data={"tool_calls": last_n_tool_calls}
                )

        return None

# monitor/detectors/progress_detector.py
class ProgressDetector(BaseDetector):
    def detect(self, state_history: List[Dict[str, Any]]) -> Optional[ProblemReport]:
        min_obs = self.config.get("min_observations", 5)
        if len(state_history) < min_obs:
            return None

        # Check for lack of output change
        max_steps_without_output_change = self.config.get("max_steps_without_output_change", 5)
        output_observations = [obs for obs in state_history if obs['type'] == 'final_output']
        
        if len(output_observations) >= max_steps_without_output_change:
            last_n_outputs = [obs['content'] for obs in output_observations[-max_steps_without_output_change:]]
            if len(set(last_n_outputs)) == 1: # All outputs are identical
                return ProblemReport(
                    problem_type="StalledOutput",
                    description=f"Agent's output has not changed for {max_steps_without_output_change} steps. Last output: '{last_n_outputs[0]}'",
                    detector_name=self.name
                )
        
        # Check for lack of thought change (more general lack of progress)
        max_steps_without_thought_change = self.config.get("max_steps_without_thought_change", 7)
        thought_observations = [obs for obs in state_history if obs['type'] == 'thought']

        if len(thought_observations) >= max_steps_without_thought_change:
            last_n_thoughts = [obs['content'] for obs in thought_observations[-max_steps_without_thought_change:]]
            if len(set(last_n_thoughts)) == 1:
                return ProblemReport(
                    problem_type="StalledThought",
                    description=f"Agent's core thought has not changed for {max_steps_without_thought_change} steps. Last thought: '{last_n_thoughts[0]}'",
                    detector_name=self.name
                )
        
        return None

# monitor/interventions/replan_intervener.py
class ReplanIntervener(BaseIntervener):
    def intervene(self, agent: BaseAgent, problem_report: ProblemReport) -> bool:
        print(f"    [Intervention] Triggering Replan for agent due to {problem_report.problem_type}.")
        try:
            agent.replan()
            agent.reset_for_replan() # Ensure agent cleans up any transient state
            return True
        except Exception as e:
            print(f"    [Intervention Error] Failed to replan agent: {e}")
            return False

# monitor/interventions/hint_intervener.py
class HintIntervener(BaseIntervener):
    def intervene(self, agent: BaseAgent, problem_report: ProblemReport) -> bool:
        hint_message = self.config.get("hint_message", "Consider re-evaluating your current approach.")
        print(f"    [Intervention] Sending hint to agent due to {problem_report.problem_type}: '{hint_message}'")
        try:
            agent.receive_hint(hint_message)
            return True
        except Exception as e:
            print(f"    [Intervention Error] Failed to send hint to agent: {e}")
            return False

# monitor/interventions/human_fallback.py
class HumanFallback(BaseIntervener):
    def intervene(self, agent: BaseAgent, problem_report: ProblemReport) -> bool:
        print(f"    [Intervention] 🚨 Human Fallback Triggered! Agent stuck due to {problem_report.problem_type}.")
        print(f"    Problem Details: {problem_report.description}")
        print("    Please review the agent's log and intervene manually.")
        # In a real system, this would trigger an alert, log to a dashboard, etc.
        # For this demo, we'll just print and stop further automated interventions for this report.
        return True # Indicate that this intervention "handled" the issue by escalating


# monitor/core.py
class MonitorCore:
    def __init__(
        self,
        state_manager: StateManager,
        instrumentation: Instrumentation, # Not directly used by monitor, but good to keep reference
        detectors: List[BaseDetector],
        interveners: List[BaseIntervener]
    ):
        self.state_manager = state_manager
        self.instrumentation = instrumentation
        self.detectors = detectors
        # Sort interveners by priority (lower number = higher priority)
        self.interveners = sorted(interveners, key=lambda x: x.config.get("priority", 99))
        self._last_problem_report: Optional[ProblemReport] = None

    def check_for_issues(self, agent: BaseAgent) -> Optional[ProblemReport]:
        current_history = self.state_manager.get_history()
        for detector in self.detectors:
            problem_report = detector.detect(current_history)
            if problem_report:
                self._last_problem_report = problem_report
                return problem_report
        self._last_problem_report = None
        return None

    def trigger_intervention(self, agent: BaseAgent, problem_report: ProblemReport) -> bool:
        for intervener in self.interveners:
            if intervener.intervene(agent, problem_report):
                # If an intervener successfully acts, we assume the issue is addressed
                # or escalated, and stop trying further interventions for this specific report.
                return True
        return False


# --- Main Scenario Runner Script ---
# examples/run_stuck_scenario.py

# Simplified config for this example (mimicking config.py)
MONITOR_CONFIG = {
    "loop_detector": {
        "thought_loop_window_size": 3,
        "tool_loop_window_size": 2,
        "min_observations": 5,
    },
    "progress_detector": {
        "max_steps_without_output_change": 5,
        "max_steps_without_thought_change": 7,
    },
    "interventions": {
        "replan_intervener": {"priority": 1},
        "hint_intervener": {"priority": 2, "hint_message": "It seems you're stuck. Consider re-evaluating your current approach or exploring new paths."},
        "human_fallback": {"priority": 3},
    },
    "monitoring_interval_steps": 1,
    "max_agent_steps": 25,
}


def run_stuck_scenario():
    print("🚀 Starting AI Agent Stuck Scenario Demonstration 🚀")
    print("-" * 60)

    # 1. Prepare Environment & Initialize Components
    state_manager = StateManager()
    instrumentation = Instrumentation(state_manager)

    task = "Find the file 'important_data.csv' in the current directory or its subdirectories. Once found, simulate reading it and report its 'content_summary'."
    
    # Create a dummy file for the agent to eventually find.
    dummy_file_dir = "data"
    dummy_file_path = os.path.join(dummy_file_dir, "important_data.csv")
    os.makedirs(dummy_file_dir, exist_ok=True)
    with open(dummy_file_path, "w") as f:
        f.write("column1,column2\nval1,val2\nA,B")
    print(f"Created dummy file: {dummy_file_path}\n")

    agent = DemoAgent(task, instrumentation=instrumentation)

    # 2. Setup Detectors
    detectors: List[BaseDetector] = [
        LoopDetector(config=MONITOR_CONFIG["loop_detector"]),
        ProgressDetector(config=MONITOR_CONFIG["progress_detector"]),
    ]

    # 3. Setup Interveners (sorted by priority during MonitorCore init)
    interveners: List[BaseIntervener] = [
        ReplanIntervener(config=MONITOR_CONFIG["interventions"]["replan_intervener"]),
        HintIntervener(config=MONITOR_CONFIG["interventions"]["hint_intervener"]),
        HumanFallback(config=MONITOR_CONFIG["interventions"]["human_fallback"]),
    ]
    
    # 4. Initialize Monitor Core
    monitor = MonitorCore(
        state_manager=state_manager,
        instrumentation=instrumentation,
        detectors=detectors,
        interveners=interveners
    )

    # 5. Run the Agent under Monitor Supervision
    print(f"Starting agent with task: '{task}'")
    print("-" * 60)

    step_count = 0
    while not agent.is_finished() and step_count < MONITOR_CONFIG["max_agent_steps"]:
        step_count += 1
        print(f"\n--- Agent Step {step_count} --- (Time: {datetime.now().strftime('%H:%M:%S')})")

        try:
            agent.step()
        except Exception as e:
            print(f"  [Agent Error] Agent encountered an unhandled exception: {e}")
            break # Exit loop on agent error

        if agent.is_finished():
            print("  [Agent] Agent reports task is finished.")
            break

        # Monitor checks for issues periodically
        if step_count % MONITOR_CONFIG["monitoring_interval_steps"] == 0:
            print(f"  [Monitor] Checking for issues...")
            problem_report = monitor.check_for_issues(agent)

            if problem_report:
                print(f"  [Monitor] 🚨 Problem Detected: {problem_report.problem_type}")
                print(f"    Details: {problem_report.description}")
                print(f"    Triggered by: {problem_report.detector_name}")

                intervened = monitor.trigger_intervention(agent, problem_report)
                if intervened:
                    print(f"  [Monitor] ✅ Intervention applied successfully.")
                else:
                    print(f"  [Monitor] ❌ No suitable intervention found or intervention failed.")
            else:
                print(f"  [Monitor] No issues detected. Agent progressing normally.")

        time.sleep(0.1) # Simulate some processing time per step

    print("\n" + "-" * 60)
    print("🏁 Scenario Finished 🏁")
    print(f"Total agent steps: {step_count}")
    print(f"Agent finished: {agent.is_finished()}")
    print("Final Agent State (last few observations from State Manager):")
    for obs in state_manager.get_history()[-5:]: # Show last 5 observations
        content_preview = str(obs.get('content', 'N/A'))[:70].replace('\n', '\\n')
        print(f"  [{obs['timestamp']}] {obs['type']}: {content_preview}...")

    # Cleanup dummy file and directory
    try:
        if os.path.exists(dummy_file_dir):
            shutil.rmtree(dummy_file_dir)
            print(f"\nCleaned up dummy directory: {dummy_file_dir}")
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    run_stuck_scenario()