This project demonstrates a Feedback-Driven Prompt Evolution System, a prototype designed to enhance the quality and relevance of Large Language Model (LLM) outputs through iterative user feedback. Static prompts often yield inconsistent or off-target results. This system addresses this by dynamically refining prompts based on user input.

The core idea is a continuous feedback loop: the system generates content using a prompt, the user provides feedback (e.g., "good", "bad", or a specific suggestion), and the system then modifies the prompt for the next iteration. For negative feedback, the LLM itself is leveraged to suggest prompt improvements. Positive feedback maintains the current prompt, while specific suggestions are integrated directly. This adaptive approach helps converge on prompts that consistently produce desired outcomes.

To use this prototype:
1. Ensure you have a Google Gemini API key.
2. Set the `GEMINI_API_KEY` environment variable (e.g., `export GEMINI_API
