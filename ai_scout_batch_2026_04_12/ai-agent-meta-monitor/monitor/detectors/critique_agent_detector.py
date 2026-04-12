```python
import json
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# Configure logging for this module
logger = logging.getLogger(__name__)
# Set default level to WARNING to avoid excessive debug/info logs during normal operation
# unless explicitly configured differently by the main application.
logger.setLevel(logging.WARNING)

# --- Assumed Base Classes and Types from Project Context ---
# In a full project, these would be imported from their respective files,
# e.g., `from monitor.detectors.base_detector import BaseDetector`
# and `from monitor.core import ProblemReport`.
# For the purpose of providing "ONLY the code for this file", we define minimal versions here.

class ProblemReport:
    """
    A standardized report for a detected problem, to be used by detectors
    to communicate issues to the Monitor Core for intervention.
    """
    def __init__(self, detector_name: str, problem_type: str, description: str,
                 severity: str, agent_id: str = "unknown_agent", additional_info: Optional[Dict[str, Any]] = None):
        self.detector_name = detector_name
        self.problem_type = problem_type
        self.description = description
        self.severity = severity
        self.agent_id = agent_id
        self.additional_info = additional_info if additional_info is not None else {}

    def __repr__(self):
        return (f"ProblemReport(detector_name='{self.detector_name}', "
                f"problem_type='{self.problem_type}', severity='{self.severity}', "
                f"description='{self.description[:70]}...', agent_id='{self.agent_id}')")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "problem_type": self.problem_type,
            "description": self.description,
            "severity": self.severity,
            "agent_id": self.agent_id,
            "additional_info": self.additional_info
        }


class BaseDetector(ABC):
    """
    Abstract base class for all detectors. All concrete detectors must inherit
    from this class and implement the `detect` method.
    """
    @abstractmethod
    def detect(self, state_history: List[Dict[str, Any]]) -> Optional[ProblemReport]:
        """
        Abstract method to detect problems in the agent's state history.

        Args:
            state_history: A list of historical states from the StateManager,
                           each representing an agent step's captured data.

        Returns:
            An optional ProblemReport if an issue is detected, otherwise None.
        """
        pass


# --- Placeholder for an LLM Client ---
# In a real scenario, this would integrate with an actual LLM provider (e.g., OpenAI, Anthropic).
# This mock client simulates an LLM's behavior for demonstration purposes.
class CritiqueLLMClient:
    """
    A mock LLM client to simulate a critique agent's response.
    In a real implementation, this would make an API call to an actual LLM.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.5, lookback_steps: int = 10):
        self.model_name = model_name
        self.temperature = temperature
        self.lookback_steps = lookback_steps  # Used by the mock for context-aware decisions
        logger.info(f"Initialized CritiqueLLMClient with model: {model_name}, temp: {temperature}, "
                    f"mock lookback_steps: {lookback_steps}")

    def generate_critique(self, prompt: str) -> str:
        """
        Simulates calling an LLM to generate a critique based on the prompt.
        This mock includes simple heuristics to demonstrate problem detection.
        """
        logger.debug(f"Calling mock LLM ({self.model_name}) with prompt snippet: {prompt[:200]}...")

        # Extract the trace content from the prompt
        trace_marker = "Agent Trace:\n"
        if trace_marker not in prompt:
            logger.warning("Mock LLM: Trace format not recognized. Cannot analyze.")
            return json.dumps({"problem_detected": False, "problem_type": None, "reasoning": "Trace not found in prompt.", "suggested_intervention": None, "relevant_trace_segment": ""})
        
        trace_content = prompt.split(trace_marker, 1)[-1]
        
        # Simple heuristics for the mock LLM to simulate detection
        steps_raw = trace_content.strip().split('--- Step ')
        # Filter out empty strings from split and prepend 'Step ' back for consistency
        steps = ["--- Step " + s.strip() for s in steps_raw if s.strip()]

        # Heuristic 1: Repetitive search tool calls with "No relevant results found"
        if "Tool Call: search_tool(" in trace_content:
            search_calls_count = trace_content.count("Tool Call: search_tool(")
            no_results_count = trace_content.count("Tool Output: No relevant results found")
            if search_calls_count >= (self.lookback_steps // 2) and no_results_count >= (self.lookback_steps // 4):
                return json.dumps({
                    "problem_detected": True,
                    "problem_type": "Repetitive Search Loop",
                    "reasoning": "The agent is repeatedly calling the search tool and frequently receiving no relevant results, indicating a loop or lack of new strategies.",
                    "suggested_intervention": "replan",
                    "relevant_trace_segment": "Repeated search calls with consistent negative outcomes."
                })
        
        # Heuristic 2: Identical consecutive steps (thought, tool, output)
        if len(steps) >= 2 and steps[-1] == steps[-2]:
            return json.dumps({
                "problem_detected": True,
                "problem_type": "Identical Step Loop",
                "reasoning": "The agent executed the exact same sequence of thought, tool call, and output multiple times consecutively.",
                "suggested_intervention": "replan_or_hint",
                "relevant_trace_segment": "Identical last two steps detected."
            })

        # Heuristic 3: Lack of significant output changes over several steps
        output_changes_count = len([line for line in trace_content.split('\n') if 'Agent Current Output:' in line])
        # If there are few distinct output lines and a good number of steps
        if output_changes_count < 2 and len(steps) >= (self.lookback_steps // 2):
            return json.dumps({
                "problem_detected": True,
                "problem_type": "Stall/Lack of Progress",
                "reasoning": "The agent has not produced any significant changes in its output over several steps, suggesting it is stalled.",
                "suggested_intervention": "hint_or_replan",
                "relevant_trace_segment": "No or minimal output changes detected."
            })
        
        # Heuristic 4: General keyword detection for looping/stalling
        if any(keyword in trace_content.lower() for keyword in ["repeatedly", "again and again", "stuck", "loop", "no progress"]):
            return json.dumps({
                "problem_detected": True,
                "problem_type": "General Repetitive/Stall Pattern",
                "reasoning": "The trace content suggests a general pattern of repetition or stalling based on keyword analysis.",
                "suggested_intervention": "replan_or_hint",
                "relevant_trace_segment": "Keyword-based detection of repetition/stall."
            })

        # Default: no problem detected by the mock
        return json.dumps({
            "problem_detected": False,
            "problem_type": None,
            "reasoning": "No apparent issues detected in the provided trace segment.",
            "suggested_intervention": None,
            "relevant_trace_segment": ""
        })


class CritiqueAgentDetector(BaseDetector):
    """
    A detector that leverages a separate LLM (a "critique agent") to analyze an
    agent's trace and identify higher-level reasoning flaws, loops, or stalls.
    This detector provides a more nuanced analysis than simple heuristic detectors.
    """
    def __init__(self,
                 llm_client: Optional[CritiqueLLMClient] = None,
                 lookback_steps: int = 10,
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.5):
        """
        Initializes the CritiqueAgentDetector.

        Args:
            llm_client: An instance of CritiqueLLMClient. If None, one will be created.
            lookback_steps: The number of recent agent steps to include in the trace for critique.
                            This determines the window of observation for the critique agent.
            model_name: The name of the LLM model to use for the critique agent (if llm_client is not provided).
            temperature: The temperature for the LLM call (if llm_client is not provided).
        """
        # Ensure the LLM client is initialized with the correct lookback_steps for better mock behavior
        self.llm_client = llm_client if llm_client else CritiqueLLMClient(
            model_name=model_name, temperature=temperature, lookback_steps=lookback_steps
        )
        self.lookback_steps = lookback_steps
        logger.info(f"CritiqueAgentDetector initialized with lookback_steps={self.lookback_steps}, "
                    f"LLM model: {self.llm_client.model_name}")

    def _format_trace_for_llm(self, trace_segment: List[Dict[str, Any]]) -> str:
        """
        Formats a segment of the agent's historical trace into a concise, readable string
        suitable for an LLM to analyze.
        Each entry in `trace_segment` is assumed to be a dictionary representing an agent step.
        """
        formatted_trace_entries = []
        for i, entry in enumerate(trace_segment):
            step_info_lines = [f"--- Step {i+1} ---"]
            
            # Agent's internal thought process
            if 'thought' in entry and entry['thought']:
                step_info_lines.append(f"Thought: {entry['thought']}")
            
            # Tool call details and its observed output
            if 'tool_name' in entry and entry['tool_name']:
                tool_args = entry.get('tool_args', '')
                tool_output = entry.get('tool_output', '')
                step_info_lines.append(f"Tool Call: {entry['tool_name']}({tool_args})")
                step_info_lines.append(f"Tool Output: {tool_output}")
            
            # Agent's final output for the step, if available.
            # The LLM can infer lack of progress if this output doesn't change over time.
            if 'output' in entry and entry['output'] is not None:
                step_info_lines.append(f"Agent Current Output: {entry['output']}")
            
            formatted_trace_entries.append("\n".join(step_info_lines))
        
        return "\n\n".join(formatted_trace_entries)

    def detect(self, state_history: List[Dict[str, Any]]) -> Optional[ProblemReport]:
        """
        Leverages an LLM (the "critique agent") to analyze a segment of the agent's trace
        and identify higher-level reasoning flaws, loops, or stalls.

        Args:
            state_history: A list of historical states from the StateManager. Each state dict is
                           expected to contain relevant keys like 'thought', 'tool_name',
                           'tool_args', 'tool_output', 'output'.

        Returns:
            An optional ProblemReport if an issue is detected, otherwise None.
        """
        # Ensure there's enough history to perform a meaningful critique
        if len(state_history) < self.lookback_steps:
            logger.debug(f"CritiqueAgentDetector: Not enough history ({len(state_history)} steps) "
                         f"for critique. Minimum {self.lookback_steps} steps needed.")
            return None

        # Extract the most recent segment of the trace for the critique agent
        trace_segment = state_history[-self.lookback_steps:]
        formatted_trace = self._format_trace_for_llm(trace_segment)

        # Construct the prompt for the critique LLM
        system_prompt = (
            "You are a meta-monitoring AI. Your task is to critically analyze the provided "
            "trace of another AI agent's execution. Identify if the agent is exhibiting "
            "problematic behavior such as repetitive loops, stalling, or logical flaws "
            "that hinder progress or lead to incorrect outcomes. "
            "Your response MUST be a JSON object with the following structure:\n"
            "{\n"
            "  \"problem_detected\": true/false,\n"
            "  \"problem_type\": \"Repetitive Loop\" | \"Stall/Lack of Progress\" | \"Reasoning Flaw\" | \"Other Problem\" | null,\n"
            "  \"reasoning\": \"A brief, clear explanation of why the problem was detected, referencing specifics from the trace.\",\n"
            "  \"suggested_intervention\": \"replan\" | \"hint\" | \"human_fallback\" | \"replan_or_hint\" | \"hint_or_replan\" | null,\n"
            "  \"relevant_trace_segment\": \"A short string or summary of the most problematic part of the trace.\"\n"
            "}"
        )

        user_prompt = (
            f"Analyze the following recent trace of an AI agent's actions and thoughts. "
            f"Pay close attention to repeating patterns, lack of progression in goals/output, "
            f"illogical decision-making, or inefficient use of tools. "
            f"If a problem is detected, provide its type, reasoning, and suggest an intervention. "
            f"If no clear problem is found, set 'problem_detected' to false.\n\n"
            f"Agent Trace:\n{formatted_trace}\n\n"
            f"Based on this trace, identify any problems and provide the critique in the specified JSON format."
        )

        try:
            # In a real LLM integration, system_prompt and user_prompt would typically be
            # sent as distinct messages in a conversation array. For this mock, we concatenate.
            full_prompt_for_mock = system_prompt + "\n\n" + user_prompt
            llm_response_str = self.llm_client.generate_critique(full_prompt_for_mock)
            
            critique_data = json.loads(llm_response_str)

            if critique_data.get("problem_detected"):
                problem_type = critique_data.get("problem_type", "Unknown Critique Problem")
                reasoning = critique_data.get("reasoning", "No specific reasoning provided by critique agent.")
                suggested_intervention = critique_data.get("suggested_intervention")
                relevant_trace = critique_data.get("relevant_trace_segment", "")

                logger.warning(f"CritiqueAgentDetector Detected Problem: Type='{problem_type}', "
                               f"Reasoning='{reasoning[:100]}...', Suggested Intervention='{suggested_intervention}'")
                
                return ProblemReport(
                    detector_name="CritiqueAgentDetector",
                    problem_type=problem_type,
                    description=reasoning,
                    severity="high", # LLM critique implies a higher-level, complex issue
                    agent_id=trace_segment[0].get('agent_id', 'unknown_agent'), # Attempt to get agent_id from trace
                    additional_info={
                        "llm_response_data": critique_data,
                        "formatted_trace_segment": formatted_trace,
                        "suggested_intervention": suggested_intervention,
                        "relevant_trace_segment": relevant_trace
                    }
                )
            else:
                logger.debug("CritiqueAgentDetector: No problems detected by the critique agent.")
                return None
        
        except json.JSONDecodeError as e:
            logger.error(f"Critique Agent LLM returned malformed JSON. Error: {e}. "
                         f"Raw Response (first 1000 chars): {llm_response_str[:1000]}...")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during critique agent LLM call: {e}", exc_info=True)
            return None

```