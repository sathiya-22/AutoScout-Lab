Problem: When interacting with large language models, especially in streaming scenarios, responses can sometimes be excessively long, consuming more tokens than desired, leading to increased costs or poor user experience. While LLMs offer `max_output_tokens` parameters, a client-side solution provides more granular control and immediate feedback, allowing applications to enforce strict token budgets dynamically.

Approach: This prototype demonstrates a "Streaming Token Budget Manager" that operates by actively monitoring the token count of a streamed LLM response. As each chunk of text arrives from the `google-genai` service, its tokens are counted. The manager maintains a running total. If adding the next incoming chunk would cause the total token count to exceed a predefined budget, the streaming process is immediately halted. This ensures that the final generated output strictly adheres to the specified token limit.

Usage:
1.  **Install dependencies:** `pip install -r requirements.txt`
2.  **Set
