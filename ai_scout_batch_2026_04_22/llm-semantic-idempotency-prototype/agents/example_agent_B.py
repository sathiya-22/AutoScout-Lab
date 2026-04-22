import logging
from typing import Any, Dict

from agents.base_agent import BaseAgent
from llm_proxies.base_proxy import BaseLLMProxy # Assuming BaseLLMProxy defines the interface for proxies

# Configure logging for this agent
logger = logging.getLogger(__name__)

class ExampleAgentB(BaseAgent):
    """
    ExampleAgentB specializes in summarizing textual content using an LLM proxy.
    It demonstrates task execution by invoking the LLM through a proxy and
    handling the output. This agent could be part of a pipeline where
    summaries are needed for further processing or reporting.
    """
    def __init__(self, name: str, llm_proxy: BaseLLMProxy):
        """
        Initializes ExampleAgentB.

        Args:
            name (str): The name of the agent.
            llm_proxy (BaseLLMProxy): An instance of an LLM proxy to interact with the LLM.
        """
        super().__init__(name, llm_proxy)
        logger.info(f"ExampleAgentB '{self.name}' initialized with proxy: {type(self.llm_proxy).__name__}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the agent's task: summarizing provided text.

        Args:
            input_data (Dict[str, Any]): A dictionary containing input data.
                                         Expected to have a 'text_to_summarize' key.

        Returns:
            Dict[str, Any]: A dictionary containing the summary or an error message.
        """
        logger.info(f"Agent '{self.name}' received input for summarization.")

        text_to_summarize = input_data.get('text_to_summarize')
        if not text_to_summarize or not isinstance(text_to_summarize, str):
            error_msg = "Invalid or missing 'text_to_summarize' in input_data."
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        prompt = (
            f"Please provide a concise summary of the following text:\n\n"
            f"TEXT:\n{text_to_summarize}\n\n"
            f"SUMMARY:"
        )

        try:
            # Invoke the LLM through the configured proxy
            llm_response = self.llm_proxy.invoke(prompt)
            summary = llm_response.strip() # Assuming the LLM returns the summary directly

            logger.info(f"Agent '{self.name}' successfully generated summary.")
            return {"status": "success", "summary": summary}
        except Exception as e:
            error_msg = f"Error during LLM invocation for summarization: {e}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}

# Example of how this agent might be instantiated and used (for internal testing/demonstration)
if __name__ == "__main__":
    # This block requires actual proxy implementations and a base agent.
    # For a full runnable example, you would need to set up the surrounding architecture.

    # Mocking BaseAgent and BaseLLMProxy for isolated testing
    # In a real scenario, these would be imported from their respective files.
    class MockBaseAgent:
        def __init__(self, name: str, llm_proxy: Any):
            self.name = name
            self.llm_proxy = llm_proxy

    class MockLLMProxy:
        def invoke(self, prompt: str) -> str:
            # Simulate an LLM response based on the prompt
            if "summarize" in prompt.lower() and "example text" in prompt.lower():
                return "This is a simulated summary of the example text."
            elif "error" in prompt.lower():
                raise ValueError("Simulated LLM error.")
            return "This is a general simulated LLM response."

    # Override the BaseAgent for testing purposes in this file
    BaseAgent = MockBaseAgent
    BaseLLMProxy = MockLLMProxy # Use the mock proxy type for type hinting consistency

    # Ensure logging is configured for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("--- Testing ExampleAgentB ---")

    mock_proxy = MockLLMProxy()
    agent_b = ExampleAgentB(name="SummarizerAgent", llm_proxy=mock_proxy)

    # Test Case 1: Successful summarization
    print("\n[Test Case 1: Successful Summarization]")
    input_text_1 = "This is an example text that needs to be summarized. It contains several sentences providing details about a hypothetical situation."
    result_1 = agent_b.run({"text_to_summarize": input_text_1})
    print(f"Agent Output 1: {result_1}")
    assert result_1["status"] == "success"
    assert "simulated summary" in result_1["summary"].lower()

    # Test Case 2: Missing input text
    print("\n[Test Case 2: Missing Input Text]")
    result_2 = agent_b.run({})
    print(f"Agent Output 2: {result_2}")
    assert result_2["status"] == "error"
    assert "missing" in result_2["message"].lower()

    # Test Case 3: Non-string input text
    print("\n[Test Case 3: Non-string Input Text]")
    result_3 = agent_b.run({"text_to_summarize": 12345})
    print(f"Agent Output 3: {result_3}")
    assert result_3["status"] == "error"
    assert "invalid" in result_3["message"].lower()

    # Test Case 4: Simulated LLM error
    print("\n[Test Case 4: Simulated LLM Error]")
    # We need a proxy that can simulate an error based on prompt for this
    class ErrorProneMockLLMProxy(MockLLMProxy):
        def invoke(self, prompt: str) -> str:
            if "error" in prompt.lower():
                raise ValueError("Simulated LLM service unavailable or bad response.")
            return super().invoke(prompt)

    error_proxy = ErrorProneMockLLMProxy()
    agent_b_error = ExampleAgentB(name="ErrorSummarizerAgent", llm_proxy=error_proxy)
    # Craft a prompt that our mock proxy will interpret as an error condition
    error_input_text = "This text should trigger an error condition during summarization."
    result_4 = agent_b_error.run({"text_to_summarize": "error " + error_input_text})
    print(f"Agent Output 4: {result_4}")
    assert result_4["status"] == "error"
    assert "error during llm invocation" in result_4["message"].lower()

    print("\n--- ExampleAgentB Tests Complete ---")