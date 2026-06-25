## Self-Healing Agent Loop with Error Recovery

### Problem
AI agents, especially those interacting with external systems or relying on complex outputs (like structured data from an LLM), are prone to failures. These can range from malformed responses and unexpected data types to logical errors or API issues. Without robust error handling, such agents can become brittle, requiring manual intervention upon failure, which hinders autonomy and scalability.

### Approach
This project demonstrates a self-healing agent loop using a "Safe Agent" and a "Recovery Agent." The Safe Agent attempts to perform a task, with built-in validation for its output. If the Safe Agent encounters an error (e.g., malformed JSON, forbidden content), the Recovery Agent is activated. The Recovery Agent, also an LLM-powered component, analyzes the error context and the original task to generate a new, refined instruction for the Safe Agent. This process allows the agent to iteratively learn from
