# Hierarchical Memory Agent Prototype

## Problem
Traditional Large Language Models (LLMs) operate with a limited context window, meaning they struggle to remember information beyond recent turns in a conversation. This limitation prevents agents from maintaining long-term coherence, recalling specific past events, or learning across interactions.

## Approach
This prototype implements a simplified Hierarchical Memory Agent. It leverages two main types of memory:
1.  **Working Memory**: Handled by the `google-genai` chat session, which maintains the immediate conversational context.
2.  **Episodic Memory**: A separate store of past `(user_message, agent_response)` pairs. When processing a new query, the agent attempts to "recall" relevant information from this episodic memory using a basic keyword matching heuristic. This recalled information is then prepended to the current prompt, providing the LLM with a richer context before generating a response. This allows the agent to retrieve and
