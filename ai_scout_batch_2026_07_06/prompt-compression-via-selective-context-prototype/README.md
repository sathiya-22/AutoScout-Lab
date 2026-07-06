This project provides a Python prototype for "Prompt Compression via Selective Context." The increasing length of user prompts and document contexts sent to Large Language Models (LLMs) can lead to higher token usage, increased latency, and potential token limit breaches. Moreover, irrelevant information in a long prompt can dilute the model's focus.

Our approach tackles this by implementing a two-stage process. First, an initial LLM call is made to a powerful model (e.g., `gemini-1.5-flash`) that acts as a "context selector." This LLM analyzes a verbose original document or conversation alongside a specific user query, extracting or summarizing only the most pertinent information relevant to that query. This results in a significantly shorter, "compressed context." Second, this optimized, concise context is then combined with the user's original query to form a new, efficient prompt. This compressed prompt is then sent to the final LLM for generating the desired response, ensuring efficiency, cost-effectiveness, and improved focus.

To use this prototype:
1. Ensure you have a Google Gemini API key. Set it as an environment variable: `export GEMINI_API_KEY='your_api_key_here'`
2. Install the necessary Python packages: `pip install -r requirements.txt`
3. Run the main script to see the demo in action: `python main.py`
The script will demonstrate how a long document is selectively compressed based on different user questions before an answer is generated.
