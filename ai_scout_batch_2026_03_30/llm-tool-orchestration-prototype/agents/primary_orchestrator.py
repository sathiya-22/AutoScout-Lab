```python
import logging
import json
from typing import Any, Dict, List, Optional

# --- Configuration for basic logging ---
# In a production system, this would be managed by a centralized config/logging setup.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MOCK/PLACEHOLDER CLASSES AND CONSTANTS ---
# In a real project, these would be imported from their respective files:
# from llm_integrations.llm_client import LLMClient
# from llm_integrations.prompt_templates import ORCHESTRATOR_PROMPT_TEMPLATE
# from llm_integrations.response_parser import ResponseParser
# from agents.tool_executor_agent import ToolExecutorAgent

class LLMClient:
    """
    A mock LLM client for demonstration purposes.
    In a real system, this would interact with an actual LLM provider.
    """
    def get_completion(self, prompt: str, **kwargs) -> str:
        logger.debug(f"MockLLMClient received prompt: {prompt[:200]}...")
        # Simulate LLM behavior based on keywords in the prompt
        # This mock output aims to demonstrate the orchestrator's decision-making flow.
        prompt_lower = prompt.lower()
        if "what is the capital of france" in prompt_lower:
            return json.dumps({
                "action": "RESPOND_TO_USER",
                "reasoning": "The user asked a factual question that can be answered directly from general knowledge.",
                "final_answer": "The capital of France is Paris."
            })
        elif "current weather in london" in prompt_lower or "get weather" in prompt_lower:
            return json.dumps({
                "action": "DELEGATE_TOOL_EXECUTION",
                "reasoning": "The user requires current weather information, which needs an external tool call.",
                "proposed_task": "Get current weather conditions",
                "parameters": {"location": "London"}
            })
        elif "deploy application" in prompt_lower or "install software" in prompt_lower:
            return json.dumps({
                "action": "DELEGATE_TOOL_EXECUTION",
                "reasoning": "The user wants to deploy an application, which requires a specialized tool.",
                "proposed_task": "Deploy a specified application",
                "parameters": {"app_name": "MyWebApp", "version": "1.0"}
            })
        elif "how do i get started" in prompt_lower or "more information" in prompt_lower:
            return json.dumps({
                "action": "AWAIT_FURTHER_INPUT",
                "reasoning": "The user's request is too vague and requires more specific input to proceed.",
                "question": "Could you please elaborate on what you would like to get started with or what information you are seeking?"
            })
        else:
            # Default to planning for more complex or unrecognized requests
            return json.dumps({
                "action": "PLAN_NEXT_STEP",
                "reasoning": "The goal is complex or unclear. I need to break it down further or identify specific sub-tasks.",
                "internal_thought": f"The current goal '{prompt_lower[:100]}...' requires further internal processing. I will re-evaluate after this thought step."
            })

class ResponseParser:
    """
    A mock response parser to convert raw LLM output (assumed JSON string) into a dictionary.
    In a real system, this would robustly handle various LLM output formats and error conditions.
    """
    def parse_orchestrator_response(self, raw_llm_output: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw_llm_output)
            # Ensure essential keys exist, provide defaults if missing
            parsed.setdefault("action", "PLAN_NEXT_STEP")
            parsed.setdefault("reasoning", "No specific reasoning provided by LLM.")
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"ResponseParser failed to parse LLM output as JSON: {e}. Raw output: {raw_llm_output}")
            # Fallback for malformed JSON or non-JSON output
            return {
                "action": "PLAN_NEXT_STEP",
                "reasoning": f"LLM output was not valid JSON or unexpected format. Error: {e}.",
                "internal_thought": "Attempting to re-plan due to unparseable LLM response."
            }
        except Exception as e:
            logger.error(f"An unexpected error occurred in ResponseParser: {e}")
            return {
                "action": "PLAN_NEXT_STEP",
                "reasoning": f"An unexpected error occurred during parsing: {e}.",
                "internal_thought": "Attempting to re-plan due to parsing error."
            }

class ToolExecutorAgent:
    """
    A mock ToolExecutorAgent for the PrimaryOrchestrator to delegate tasks to.
    In a real system, this agent would be fully implemented in agents/tool_executor_agent.py,
    handling tool selection, verification (via VerifierAgent), and sandboxed execution.
    """
    def execute_tool_task(self, task_description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"MockToolExecutorAgent received task: '{task_description}' with parameters: {parameters}")
        # Simulate different outcomes based on task_description
        if "weather" in task_description.lower():
            location = parameters.get("location", "unknown location")
            logger.info(f"Simulating getting weather for {location}...")
            return {"status": "success", "output": f"The current weather in {location} is sunny with 25°C."}
        elif "deploy" in task_description.lower() or "install" in task_description.lower():
            app_name = parameters.get("app_name", "unknown application")
            logger.info(f"Simulating deployment of {app_name}...")
            # Simulate a scenario where deployment might sometimes fail
            if app_name == "FaultyApp":
                return {"status": "error", "error": f"Deployment of {app_name} failed due to dependency issues."}
            return {"status": "success", "output": f"Application '{app_name}' deployed successfully."}
        else:
            logger.warning(f"MockToolExecutorAgent: Unrecognized task '{task_description}'. Simulating failure.")
            return {"status": "error", "error": f"Tool execution failed: unrecognized task '{task_description}'."}

# Placeholder for the orchestrator's prompt template.
# In a real system, this would be defined in llm_integrations/prompt_templates.py
ORCHESTRATOR_PROMPT_TEMPLATE = """
You are the Primary Orchestrator Agent. Your top-level responsibility is to interpret high-level user goals, break them down into actionable sub-tasks, and intelligently delegate these to specialized sub-agents. You must manage the overall workflow state, orchestrate multi-step reasoning processes, and ensure progress towards the user's objective.

**Current User Goal**: {user_goal}

**Conversation History and Workflow State**:
{conversation_history}

Based on the current user goal and the history of interactions and internal thoughts, determine the single most appropriate next step. Your output MUST be a JSON object with an "action" field and other relevant fields as described below. If you cannot decide, default to 'PLAN_NEXT_STEP'.

**Available Actions and their required fields**:

1.  **DELEGATE_TOOL_EXECUTION**: Use this when a specific external tool or a sub-agent dedicated to tool execution (like the ToolExecutorAgent) is clearly needed to achieve a part of the goal.
    *   **reasoning**: (string) A concise explanation of why delegating tool execution is the next logical step.
    *   **proposed_task**: (string) A clear, actionable description of the task for the Tool Executor Agent. This should be specific enough for the Tool Executor to understand what to do (e.g., "Get current stock price for Apple", "Send email to user 'john.doe@example.com' with subject 'Meeting Reminder' and body 'Don't forget the meeting at 3 PM'").
    *   **parameters**: (object) A dictionary of key-value pairs representing any specific arguments or context the Tool Executor Agent might need for the proposed task (e.g., {{ "symbol": "AAPL", "recipient": "john.doe@example.com", "subject": "Meeting Reminder" }}).

2.  **RESPOND_TO_USER**: Use this when the user's goal has been fully achieved, requires a direct answer based on current information, or cannot be processed further by the system. This action concludes the current orchestration cycle.
    *   **reasoning**: (string) Explain why you are responding to the user and how the goal has been addressed.
    *   **final_answer**: (string) The complete, clear, and concise answer or response to the user.

3.  **AWAIT_FURTHER_INPUT**: Use this when the current user goal is ambiguous, incomplete, or requires more specific information from the user before any further action can be taken.
    *   **reasoning**: (string) Explain why more input is needed from the user.
    *   **question**: (string) The specific, clear question to ask the user to gather the necessary information.

4.  **PLAN_NEXT_STEP**: Use this when further internal reasoning, decomposition of the goal into smaller sub-problems, or preparation is required before you can delegate, respond, or ask for more input. This action means you are not yet ready for an external interaction.
    *   **reasoning**: (string) Explain the purpose of this internal planning step.
    *   **internal_thought**: (string) Detail your thought process, what you are trying to figure out, or the next internal logical step.

**Important**: Your output must always be a valid JSON object. Do not include any text before or after the JSON.

Your response:
"""
# --- END MOCK/PLACEHOLDER CLASSES AND CONSTANTS ---


class PrimaryOrchestrator:
    """
    The top-level agent responsible for interpreting high-level user goals,
    breaking them down into sub-tasks, and delegating to specialized sub-agents.
    It manages the overall workflow state and orchestrates multi-step reasoning processes.
    """
    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor_agent: ToolExecutorAgent,
        response_parser: ResponseParser
    ):
        """
        Initializes the PrimaryOrchestrator.

        Args:
            llm_client: An instance of LLMClient to interact with the language model.
            tool_executor_agent: An instance of ToolExecutorAgent for delegating tool calls.
            response_parser: An instance of ResponseParser to parse LLM outputs.
        """
        self.llm_client = llm_client
        self.tool_executor_agent = tool_executor_agent
        self.response_parser = response_parser
        self.conversation_history: List[Dict[str, Any]] = []
        logger.info("PrimaryOrchestrator initialized successfully.")

    def _update_history(self, role: str, content: Any):
        """
        Updates the internal conversation history with a new entry.

        Args:
            role: The role associated with the content (e.g., "user", "orchestrator_thought", "tool_executor_result").
            content: The actual content to add to the history.
        """
        entry = {"role": role, "content": content}
        self.conversation_history.append(entry)
        logger.debug(f"History updated: {entry}")

    def _get_orchestrator_prompt(self, user_goal: str) -> str:
        """
        Constructs the prompt for the orchestrator's LLM, incorporating the user goal
        and the current conversation history.

        Args:
            user_goal: The high-level goal provided by the user.

        Returns:
            A formatted string ready to be sent to the LLM.
        """
        # Format the history for inclusion in the prompt
        history_str = "\n".join([f"[{item['role']}]: {item['content']}" for item in self.conversation_history])

        # Use the predefined template to construct the full prompt
        return ORCHESTRATOR_PROMPT_TEMPLATE.format(
            user_goal=user_goal,
            conversation_history=history_str if history_str else "No prior history."
        )

    def run(self, user_goal: str) -> Dict[str, Any]:
        """
        Executes the primary orchestration workflow for a given high-level user goal.
        This method manages the multi-step reasoning, delegation, and state updates.

        Args:
            user_goal: The initial high-level goal provided by the user.

        Returns:
            A dictionary containing the final status and output/error/question.
            Example: {"status": "success", "output": "..."}
                     {"status": "error", "message": "..."}
                     {"status": "requires_input", "question": "..."}
        """
        logger.info(f"Orchestrator starting workflow for user goal: '{user_goal}'")
        self._update_history("user", user_goal)

        max_iterations = 15  # Safety guard to prevent infinite loops in complex workflows
        current_iteration = 0

        while current_iteration < max_iterations:
            current_iteration += 1
            logger.debug(f"Orchestration Loop Iteration: {current_iteration}")

            prompt = self._get_orchestrator_prompt(user_goal)
            logger.debug(f"Orchestrator sending prompt to LLM (Iteration {current_iteration}): {prompt}")

            try:
                # 1. Get LLM's decision on the next action
                llm_raw_response = self.llm_client.get_completion(prompt)
                logger.debug(f"Orchestrator received raw LLM response: {llm_raw_response}")

                # 2. Parse the LLM's structured response
                parsed_response = self.response_parser.parse_orchestrator_response(llm_raw_response)
                action = parsed_response.get("action")
                reasoning = parsed_response.get("reasoning", "No reasoning provided by LLM.")
                logger.info(f"Orchestrator decision (Iteration {current_iteration}): Action='{action}', Reasoning='{reasoning}'")
                self._update_history("orchestrator_thought", {"action": action, "reasoning": reasoning})

                # 3. Execute action based on LLM's decision
                if action == "DELEGATE_TOOL_EXECUTION":
                    proposed_task = parsed_response.get("proposed_task")
                    parameters = parsed_response.get("parameters", {})

                    if not proposed_task:
                        logger.error("DELEGATE_TOOL_EXECUTION action requires a 'proposed_task'.")
                        self._update_history("error", "Orchestrator failed to specify a task for tool execution.")
                        return {"status": "error", "message": "Orchestrator internal error: 'proposed_task' missing for tool delegation."}

                    logger.info(f"Delegating task to ToolExecutorAgent: '{proposed_task}' with params: {parameters}")
                    tool_result = self.tool_executor_agent.execute_tool_task(proposed_task, parameters)
                    self._update_history("tool_executor_result", tool_result)

                    if tool_result.get("status") == "success":
                        logger.info(f"Tool execution successful. Output: {tool_result.get('output')}")
                        # Continue the loop; orchestrator re-evaluates next step based on tool's output
                    else:
                        error_message = tool_result.get("error", "Unknown tool execution error.")
                        logger.error(f"Tool execution failed. Error: {error_message}")
                        self._update_history("error", f"Tool execution failed: {error_message}")
                        return {"status": "error", "message": f"Tool execution failed: {error_message}"}

                elif action == "RESPOND_TO_USER":
                    final_answer = parsed_response.get("final_answer")
                    if not final_answer:
                        logger.error("RESPOND_TO_USER action requires 'final_answer'.")
                        return {"status": "error", "message": "Orchestrator internal error: 'final_answer' missing for user response."}
                    logger.info(f"Orchestrator concluding workflow and responding to user: {final_answer}")
                    return {"status": "success", "output": final_answer}

                elif action == "AWAIT_FURTHER_INPUT":
                    question = parsed_response.get("question")
                    if not question:
                        logger.error("AWAIT_FURTHER_INPUT action requires 'question'.")
                        return {"status": "error", "message": "Orchestrator internal error: 'question' missing for user input request."}
                    logger.info(f"Orchestrator pausing workflow, awaiting further input: {question}")
                    return {"status": "requires_input", "question": question}

                elif action == "PLAN_NEXT_STEP":
                    internal_thought = parsed_response.get("internal_thought", reasoning)
                    logger.info(f"Orchestrator internally planning next step: {internal_thought}")
                    # History is already updated with orchestrator_thought, loop continues to re-evaluate
                else:
                    logger.warning(f"Orchestrator received an unrecognized action '{action}' from LLM. Attempting to re-plan.")
                    self._update_history("warning", f"LLM returned an unrecognized action '{action}'. Re-evaluating strategy.")
                    # The loop will naturally continue, prompting the LLM again with the updated history.

            except Exception as e:
                logger.exception(f"An unexpected error occurred during orchestration process for goal '{user_goal}'.")
                self._update_history("error", f"Orchestrator encountered an unexpected error: {str(e)}")
                return {"status": "error", "message": f"Orchestrator failed due to an unexpected internal error: {str(e)}"}

        logger.warning(f"Orchestrator reached maximum iterations ({max_iterations}) for goal '{user_goal}' without concluding.")
        return {"status": "error", "message": f"Orchestrator reached maximum iterations ({max_iterations}) and could not complete the goal."}

if __name__ == "__main__":
    # Example usage with mock classes
    logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logs in example

    llm_client = LLMClient()
    tool_executor_agent = ToolExecutorAgent()
    response_parser = ResponseParser()

    orchestrator = PrimaryOrchestrator(llm_client, tool_executor_agent, response_parser)

    print("\n--- Running Orchestrator for 'What is the capital of France?' ---")
    result1 = orchestrator.run("What is the capital of France?")
    print(f"Final Result 1: {result1}")

    print("\n--- Running Orchestrator for 'Get current weather in London' ---")
    result2 = orchestrator.run("Get current weather in London")
    print(f"Final Result 2: {result2}")

    print("\n--- Running Orchestrator for 'Deploy MyWebApp' ---")
    result3 = orchestrator.run("Deploy MyWebApp")
    print(f"Final Result 3: {result3}")

    print("\n--- Running Orchestrator for 'I need help' (ambiguous) ---")
    result4 = orchestrator.run("I need help")
    print(f"Final Result 4: {result4}")

    print("\n--- Running Orchestrator for 'Analyze sales data' (complex, likely PLAN_NEXT_STEP) ---")
    result5 = orchestrator.run("Analyze sales data for Q3 2023 and identify top 5 performing products.")
    print(f"Final Result 5: {result5}")

    print("\n--- Running Orchestrator for 'Deploy FaultyApp' (simulating tool failure) ---")
    result6 = orchestrator.run("Deploy FaultyApp")
    print(f"Final Result 6: {result6}")
```