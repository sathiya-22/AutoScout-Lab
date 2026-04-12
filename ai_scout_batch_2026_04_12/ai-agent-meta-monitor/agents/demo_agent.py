```python
import time
import random
from .base_agent import BaseAgent

class DemoAgent(BaseAgent):
    """
    A concrete implementation of a simple agent designed to sometimes
    exhibit repetitive looping or stalling behavior to stress-test the monitor.

    The agent's task is to "Fetch a specific data item". It will repeatedly
    try to fetch `LOOP_TRIGGER_ITEM`, which is simulated to fail a few times
    before succeeding or being interrupted by the monitor.
    """
    LOOP_TRIGGER_ITEM = "critical_data_A"
    MAX_LOOP_ATTEMPTS = 5 # How many times it tries to fetch critical_data_A before succeeding normally
    LOOP_BREAK_ITEM = "alternative_data_B" # Item to try after intervention

    def __init__(self, name: str, task: str):
        super().__init__(name, task)
        self.step_count = 0
        self.internal_state = {
            "current_goal": "fetch_initial_data",
            "data_item_to_fetch": self.LOOP_TRIGGER_ITEM,
            "fetch_attempts": 0,
            "last_tool_result": None,
            "strategy": "direct_fetch",
            "wait_count": 0 # For 'wait_and_retry' strategy
        }
        self.has_looped_once = False # Flag to indicate if the loop behavior has been triggered
        self.has_received_hint = False
        self.has_replanned = False

    def run(self) -> str:
        """
        Executes the agent's task. This method includes the logic to
        demonstrate looping behavior under specific conditions.
        """
        self.is_running = True
        final_output = ""
        print(f"\n--- [{self.name}] Starting task: {self.task} ---")

        while self.is_running and self.step_count < 25: # Limit total steps for demo
            self.step_count += 1
            current_goal = self.internal_state["current_goal"]
            data_item = self.internal_state["data_item_to_fetch"]
            strategy = self.internal_state["strategy"]

            print(f"[{self.name}][Step {self.step_count}] Goal='{current_goal}', Data='{data_item}', Strategy='{strategy}'")

            try:
                if current_goal == "fetch_initial_data":
                    if strategy == "wait_and_retry" and self.internal_state["wait_count"] < 2:
                        thought = self._generate_thought(
                            f"Strategy is '{strategy}'. Waiting for a bit before retrying fetch of '{data_item}'."
                        )
                        print(f"[{self.name}] {thought}")
                        self.internal_state["wait_count"] += 1
                        time.sleep(0.5) # Simulate a longer wait
                        continue # Skip to next loop iteration after waiting

                    thought = self._generate_thought(
                        f"My current goal is to {current_goal}. I need to fetch '{data_item}' using a '{strategy}' strategy."
                    )
                    print(f"[{self.name}] {thought}")

                    tool_name = "api_data_service"
                    tool_args = {"item": data_item, "strategy": strategy}

                    tool_result = self._simulated_api_fetch(tool_name, **tool_args)
                    self.internal_state["last_tool_result"] = tool_result
                    processed_result = self._process_tool_result(tool_result)
                    print(f"[{self.name}] {processed_result}")

                    if tool_result.get("status") == "success":
                        self.internal_state["current_goal"] = "process_data"
                        self.internal_state["fetched_data"] = tool_result.get("data")
                        print(f"[{self.name}] Successfully fetched data: {tool_result.get('data')}")
                    else:
                        print(f"[{self.name}] Failed to fetch data: {tool_result.get('message')}")
                        if data_item == self.LOOP_TRIGGER_ITEM and strategy == "direct_fetch":
                            self.internal_state["fetch_attempts"] += 1
                            print(f"[{self.name}] Fetch attempts for {self.LOOP_TRIGGER_ITEM}: {self.internal_state['fetch_attempts']}")
                            # Agent naturally loops here by not changing its goal, waiting for success or intervention
                        elif self.has_replanned and strategy != "give_up":
                            print(f"[{self.name}] Replanned strategy '{strategy}' failed. Considering fallback.")
                            # If replan failed, switch to a definitive failure state
                            self.internal_state["current_goal"] = "conclude_failure"
                        else:
                            # For other failures, or if already tried replanning and it failed, give up.
                            self.internal_state["current_goal"] = "conclude_failure"


                elif current_goal == "process_data":
                    data = self.internal_state.get("fetched_data")
                    if not data:
                        raise ValueError("No data found to process despite success status.")
                    thought = self._generate_thought(f"Data '{data_item}' fetched. Now processing: {data}")
                    print(f"[{self.name}] {thought}")
                    final_output = self._produce_output(f"Task completed successfully. Processed '{data_item}': {data}")
                    print(f"[{self.name}] {final_output}")
                    return final_output # Exit run loop

                elif current_goal == "conclude_failure":
                    final_output = self._produce_output(f"Task failed after multiple attempts and interventions. Goal: {self.task}")
                    print(f"[{self.name}] {final_output}")
                    return final_output

                else:
                    print(f"[{self.name}] Unknown goal state: {current_goal}. Terminating.")
                    final_output = self._produce_output(f"Agent entered unknown state: {current_goal}")
                    return final_output

            except Exception as e:
                print(f"[{self.name}] An error occurred during step {self.step_count}: {e}")
                final_output = self._produce_output(f"Task aborted due to error: {e}")
                return final_output

            time.sleep(0.2) # Simulate work between steps to make observation traces clearer

        if self.is_running: # If loop exited due to step limit
            print(f"[{self.name}] Step limit reached ({self.step_count}). Force stopping.")
            final_output = self._produce_output(f"Task stopped due to step limit. Last goal: {current_goal}")
        return final_output

    def _simulated_api_fetch(self, tool_name: str, **kwargs) -> dict:
        """
        Simulates an API call that fails for a specific item multiple times
        to trigger a loop.
        """
        item = kwargs.get("item")
        strategy = kwargs.get("strategy")

        self.current_tool_call = (tool_name, kwargs)
        time.sleep(0.1) # Simulate network latency

        if item == self.LOOP_TRIGGER_ITEM and strategy == "direct_fetch":
            # Introduce the loop behavior: fail for MAX_LOOP_ATTEMPTS - 1 times
            current_attempts = self.internal_state["fetch_attempts"]
            if current_attempts < self.MAX_LOOP_ATTEMPTS - 1:
                self.has_looped_once = True
                return {"status": "failure", "message": f"API busy for '{item}', retry later. (Attempt {current_attempts + 1})"}
            else:
                # Succeed after MAX_LOOP_ATTEMPTS if no intervention occurred
                return {"status": "success", "data": f"Data_content_for_{item} (from direct fetch)"}

        elif item == self.LOOP_BREAK_ITEM and strategy == "alternative_source":
            # This is the path taken after an intervention.
            # Introduce a small chance of failure even for alternative sources to stress the system more.
            if random.random() < 0.1 and not self.has_received_hint:
                 return {"status": "failure", "message": f"Alternative source for '{item}' also busy. Consider another option."}
            return {"status": "success", "data": f"Data_content_for_{item} (from alternative source)"}

        elif strategy == "wait_and_retry":
            # If this strategy is active, it means the agent has already waited, so now it succeeds.
            # In a more complex demo, this might still have a failure chance.
            return {"status": "success", "data": f"Data_content_for_{item} (after waiting)"}

        else:
            # Default success for other items or strategies not specifically designed to loop
            return {"status": "success", "data": f"Generic_data_content_for_{item}"}

    def replan(self, problem_description: str):
        """
        Intervention: The agent is instructed to re-plan its strategy.
        It changes its approach and target data item to break out of a loop.
        """
        print(f"\n--- [{self.name}] INTERVENTION: REPLAN initiated due to: {problem_description} ---")
        self.current_thought = f"Received replan request: {problem_description}. Adjusting strategy."
        self.internal_state["strategy"] = "alternative_source"
        self.internal_state["data_item_to_fetch"] = self.LOOP_BREAK_ITEM # Try a different item
        self.internal_state["fetch_attempts"] = 0 # Reset attempts for new strategy
        self.internal_state["wait_count"] = 0 # Reset wait state
        self.internal_state["current_goal"] = "fetch_initial_data" # Go back to fetching with new params
        self.has_replanned = True
        print(f"[{self.name}] Re-planned! New strategy: '{self.internal_state['strategy']}', new item: '{self.internal_state['data_item_to_fetch']}'")
        self.step_count += 1 # Count intervention as a step

    def receive_hint(self, hint: str):
        """
        Intervention: The agent receives a contextual hint. It might adjust
        its strategy or target based on the hint.
        """
        print(f"\n--- [{self.name}] INTERVENTION: HINT received: {hint} ---")
        self.current_thought = f"Received hint: {hint}. Incorporating into plan."
        self.has_received_hint = True
        if "try alternative" in hint.lower():
            self.internal_state["strategy"] = "alternative_source"
            self.internal_state["data_item_to_fetch"] = self.LOOP_BREAK_ITEM
            self.internal_state["fetch_attempts"] = 0
            self.internal_state["wait_count"] = 0
            self.internal_state["current_goal"] = "fetch_initial_data"
            print(f"[{self.name}] Hint applied. Switching to alternative source for '{self.LOOP_BREAK_ITEM}'.")
        elif "wait before retry" in hint.lower():
            self.internal_state["strategy"] = "wait_and_retry"
            self.internal_state["fetch_attempts"] = 0 # Reset attempts count
            self.internal_state["wait_count"] = 0 # Start new wait cycle
            self.internal_state["current_goal"] = "fetch_initial_data" # Re-attempt current item with wait strategy
            print(f"[{self.name}] Hint applied. Will 'wait_and_retry' for '{self.internal_state['data_item_to_fetch']}'.")
        else:
            print(f"[{self.name}] Hint not directly actionable by specific logic, but noted: {hint}")
        self.step_count += 1 # Count intervention as a step
```