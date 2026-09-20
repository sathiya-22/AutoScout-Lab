# Major problem of week 2026-W38

**Agent portability across frameworks**  (id: `agent-portability-across-frameworks`, signal: 216)

Developers struggle to make AI chat sessions and agentic capabilities portable across different agent frameworks and platforms. This limits interoperability and reusability, forcing developers to rebuild or adapt agents for each new environment.

## Why this one

The problem of agent portability across frameworks affects a vast number of developers, as the agentic AI landscape is highly fragmented with numerous competing frameworks. The severity is high, as it forces significant re-engineering effort and limits the adoption of agentic solutions due to vendor lock-in and lack of interoperability. While a prototype exists, a robust, widely adopted open-source solution is still absent, making it a prime candidate for a serious community effort. Its feasibility for a small project is also good, as initial efforts can focus on a common interchange format and a few key framework adapters.

## Sources

- https://news.ycombinator.com/item?id=49743049

Daily prototype: https://github.com/sathiya-22/agent-portability-across-frameworks-2026-09-18

---

## Problem
Developers struggle to make AI chat sessions and agentic capabilities portable across different agent frameworks and platforms. This limits interoperability and reusability, forcing developers to rebuild or adapt agents for each new environment. The current landscape of agentic AI frameworks (e.g., LangChain, AutoGen, CrewAI, LlamaIndex, Model Context Protocol) is highly fragmented, leading to significant friction when attempting to migrate or integrate agents built with one framework into another.

## Evidence
- **Signal:** 216 upvotes/reactions on a Hacker News discussion, indicating widespread community pain.
- **Status:** 'prototyped', suggesting initial attempts but no widely adopted, robust solution.
- **Prototype Repo:** `https://github.com/sathiya-22/agent-portability-across-frameworks-2026-09-18` confirms that developers are actively trying to solve this, but a community-driven, framework-agnostic standard is still missing.
- **Implicit evidence:** The very existence and rapid evolution of multiple agent frameworks inherently creates this portability challenge.

## Proposed solution
We propose an open-source project called `AgentBridge` that provides a standardized, framework-agnostic interchange format for agentic components and session states, along with a set of adapters for popular agent frameworks. The core idea is to define a minimal, yet expressive, schema for representing agent definitions (tools, LLM configurations, memory types, system prompts) and conversation histories/session states. This would allow developers to export an agent or a session from one framework and import it into another, or even run it with a lightweight, framework-agnostic runtime.

## MVP scope
1.  **Core Schema Definition:** Define a JSON-based schema for representing:
    *   Basic agent configuration (LLM model name, temperature, system prompt).
    *   Tool definitions (name, description, input schema).
    *   Conversation history (list of messages with role, content, and optional tool calls/results).
2.  **LangChain Adapter:** Implement a module to:
    *   Export a simple LangChain `Runnable` agent (with tools) and its `ChatMessageHistory` into the `AgentBridge` schema.
    *   Import an `AgentBridge` schema into a basic LangChain `Runnable` agent.
3.  **Basic CLI Tool:** A command-line interface to:
    *   `agentbridge export --framework langchain --agent-path my_agent.py --output agent.json`
    *   `agentbridge import --framework autogen --input agent.json --output my_autogen_agent.py` (initially just print the equivalent AutoGen config).
4.  **Documentation:** Clear documentation on the schema, usage, and how to contribute new adapters.

## Milestones
### Milestone 1: Core Schema & LangChain Export (2 weeks)
*   Define initial `AgentBridge` JSON schema for agent config and chat history.
*   Implement `AgentBridge.export_langchain()` function to convert a simple LangChain agent and its history to the schema.
*   Unit tests for schema validation and LangChain export.

### Milestone 2: LangChain Import & Basic CLI (2 weeks)
*   Implement `AgentBridge.import_langchain()` function to convert the schema back to a basic LangChain agent.
*   Develop a basic CLI tool for `export` and `import` operations with LangChain.
*   End-to-end integration tests for LangChain export/import via CLI.

### Milestone 3: AutoGen Adapter (3 weeks)
*   Implement `AgentBridge.export_autogen()` for a basic AutoGen agent.
*   Implement `AgentBridge.import_autogen()` for a basic AutoGen agent.
*   Update CLI to support AutoGen framework.
*   Comprehensive documentation for the `AgentBridge` schema and how to add new framework adapters.

### Milestone 4: Community & Refinement (Ongoing)
*   Gather feedback from early adopters and refine the schema and adapters.
*   Add support for more complex agent features (e.g., memory types, custom callbacks).
*   Integrate with other popular frameworks (e.g., CrewAI, LlamaIndex, MCP).
