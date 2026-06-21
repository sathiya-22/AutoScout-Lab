This project provides a Python prototype for a Task Decomposition and Planning Agent. The agent leverages Google's Gemini LLM to break down complex, high-level goals into manageable sub-tasks and then generates a detailed execution plan for each individual step.

**Problem:** Large Language Models (LLMs) can struggle with complex, multi-faceted tasks when given a single prompt. Decomposing a task into smaller, sequential steps greatly improves an LLM's ability to generate coherent and actionable responses.

**Approach:**
1.  **Task Decomposition:** The agent first takes a high-level task and uses the LLM to generate a numbered list of atomic, actionable steps.
2.  **Step-by-Step Planning:** For each identified step, the agent then prompts the LLM again to create a detailed, practical plan to accomplish that specific sub-task.

**Usage:**
1.  **Set up your environment:** Ensure you have Python 3.9+ installed.
2.  **Install dependencies:** `pip install -r requirements.txt`
3.  **Configure API Key:** Obtain a Google Gemini API key and set it as an environment variable named `GEMINI_API_KEY`. You can also create a `.env` file in the project root with the line `GEMINI_API_KEY="YOUR_API_KEY_HERE"`.
4.  **Run the agent:** `python main.py`
5.  Follow the prompts to enter a high-level task. The agent will output the decomposed steps and their respective plans.
