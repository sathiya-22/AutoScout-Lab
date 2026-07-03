import os
import google.generativeai as genai
from config import settings

def main():
    # Configure the generative AI model using settings from config.py
    genai.configure(api_key=settings.api_key)

    model = genai.GenerativeModel(
        model_name=settings.model_name,
        generation_config={
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_tokens,
        },
    )

    # --- Sample Scenario for Verification ---
    # A user query that requires tool use.
    user_query = "What is the current temperature in London and what time is it there?"

    # An AI's proposed Chain-of-Thought, including tool calls and observations.
    # This example includes a deliberate error to showcase the verifier's capability.
    ai_chain_of_thought = """
    Thought: The user wants to know the current temperature and time in London.
    I should first find the current weather in London to get the temperature.
    Then, I need to find the current time in London.

    Tool Call: call_weather_api(location="London")
    Observation: {"temperature": "15°C", "conditions": "Cloudy"}

    Thought: I have the temperature. Now I need the time.
    I will try to use the weather API again, asking for time.

    Tool Call: call_weather_api(location="London", get_time=True) # <-- This is a flawed tool call
    Observation: Error: 'get_time' is not a valid parameter for weather_api. The weather API only provides weather data.

    Thought: The weather API does not provide time as indicated by the error. I need a separate tool for time.
    I should call a dedicated time API.

    Tool Call: call_time_api(location="London")
    Observation: {"time": "10:30 AM", "timezone": "GMT"}

    Final Answer: The current temperature in London is 15°C and the time is 10:30 AM GMT.
    """

    # --- Construct the Verifier Prompt ---
    # This prompt instructs the LLM to act as a verifier and analyze the CoT.
    verifier_prompt = f"""
    You are an AI Chain-of-Thought Verifier. Your task is to critically analyze an AI's reasoning process,
    including its tool calls and observations, against a given user query.

    Evaluate the following aspects:
    1.  **Logical Flow:** Is the reasoning coherent and does it logically address the user's query?
    2.  **Tool Selection & Parameters:** Are the chosen tools appropriate for the task? Are the parameters
        passed to the tools correct and valid according to the tool's likely functionality?
    3.  **Observation Interpretation:** Is the AI correctly interpreting the observations returned by the tools?
    4.  **Error Handling:** If an error occurs, does the AI recover appropriately and learn from the error?
    5.  **Final Answer:** Does the final answer accurately and completely address the user's query
        based on the chain of thought?

    ---
    User Query: {user_query}
    ---
    AI's Chain-of-Thought:
    {ai_chain_of_thought}
    ---

    Provide a detailed verification report, highlighting any strengths, weaknesses, or potential improvements.
    """

    print("--- Verifier Analyzing Chain-of-Thought ---")
    try:
        response = model.generate_content(verifier_prompt)
        print(response.text)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your GEMINI_API_KEY environment variable is correctly set and is valid.")

if __name__ == "__main__":
    main()
