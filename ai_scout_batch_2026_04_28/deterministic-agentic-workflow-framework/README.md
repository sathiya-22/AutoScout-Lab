# Deterministic Agentic Workflow Framework

## Introduction: The Semantic Idempotency Problem in Multi-Agent AI

In the evolving landscape of multi-agentic AI systems, even robust infrastructure-level retry mechanisms fall short in addressing a fundamental challenge: the "semantic idempotency" problem. The non-deterministic nature of Large Language Model (LLM) outputs means that re-invoking an LLM after a failure often produces a *different* output, rather than an semantically equivalent one. This leads to classic byzantine failures, where individual agents may hold inconsistent views of the system state or task progress, severely compromising system reliability.

Such inherent unpredictability makes robust error handling, state reconciliation, and workflow reliability extremely difficult, often requiring extensive, costly, and manual external validation to ensure coherence and correctness. This framework aims to solve this challenge.

## Solution Overview: Towards Predictable Agentic Workflows

The "Deterministic Agentic Workflow Framework" is designed to mitigate the "semantic idempotency" problem by introducing layers of robust state management, semantic verification, and self-correction. It transforms byzantine failures (where different agents perceive different system states) into more tractable crash failures (where a failure is clearly detected, and the system can recover).

This framework provides maximum observability and control over non-deterministic LLM interactions, allowing for the construction of reliable and auditable multi-agent systems.

## Core Principles

1.  **State Centralization & Formalization:** All agent actions and system states are formally defined, strictly schema-validated, and managed centrally. This ensures a single source of truth.
2.  **Semantic Observability:** The system moves beyond mere string comparison to understand the "meaning" or semantic equivalence of LLM outputs, crucial for detecting true inconsistencies.
3.  **Recoverability & Rollback:** The ability to revert to known good, consistent states upon detection of any inconsistency or error, ensuring workflow resilience.
4.  **Verifiability:** Integration with formal rules and invariants to automatically generate tests and validate adherence to predefined business logic and expected outcomes.

## Architecture & Key Components

The framework is structured into several interconnected components, each playing a critical role in achieving determinism and reliability:

### 1. `framework/` (Core Workflow & State Management)

The foundational layer for orchestrating workflows and managing system state.

*   **`state_manager.py`**: The heart of the framework. It maintains a canonical, formalized representation of the system's state and a comprehensive history of all agent actions. All state mutations are processed through this manager, ensuring atomic transactions, checkpointing, and rollback functionality. States and actions are defined using Pydantic models (from `action_schemas.py`) for strict schema enforcement and validation.
*   **`action_schemas.py`**: Defines structured Pydantic models for agent inputs, outputs, proposed actions, and their expected semantic outcomes. This formalization is critical for semantic comparison, state validation, and effective reconciliation.
*   **`workflow_manager.py`**: Orchestrates the execution of agent workflows. It interacts deeply with the `StateManager` to record each step, request checkpoints, and initiate rollbacks when necessary. It manages the sequence, dependencies, and execution flow of agent tasks.
*   **`exceptions.py`**: Custom exceptions for framework-specific error handling, enabling clearer error reporting and recovery strategies.

### 2. `agents/` (Agent Definitions)

Defines the structure and interface for agents operating within the framework.

*   **`base_agent.py`**: An abstract base class that all operational agents must inherit from. It enforces a standardized interface, ensuring agents register their proposed actions and expected outcomes with the `StateManager` (via `WorkflowManager`) and interact with LLMs exclusively through `utils/llm_connector.py`.
*   **`example_data_processor_agent.py`**: A concrete demonstration of how to implement an agent according to the framework's guidelines, performing a task that might involve non-deterministic LLM calls.

### 3. `semantic_comparison/` (Output Equivalence & Divergence Detection)

Modules dedicated to understanding the "meaning" of LLM outputs beyond surface-level string matching.

*   **`comparators.py`**: Implements various sophisticated strategies for comparing LLM outputs semantically:
    *   **Embedding-based similarity:** Utilizes vector embeddings (e.g., from OpenAI, Sentence-BERT) to quantify semantic proximity between different outputs.
    *   **Structured Diffing:** For outputs conforming to predefined schemas (e.g., JSON/YAML), it compares parsed structures, intelligently ignoring cosmetic differences while flagging substantive ones.
    *   **LLM-as-Comparator:** Leverages an LLM itself to evaluate two distinct outputs for semantic equivalence or divergence within a specific task context.
*   **`evaluation_metrics.py`**: Defines metrics and thresholds used by the comparators to determine objective equivalence or significant divergence (e.g., cosine similarity thresholds, structural consistency scores, LLM-generated confidence scores).

### 4. `reconciliation/` (Self-Correction & Inconsistency Resolution)

Specialized intelligence for resolving detected discrepancies and guiding the system back to a consistent state.

*   **`reconciliation_agent.py`**: A specialized, often meta-agentic component responsible for observing inconsistencies flagged by the `WorkflowManager` or `Verifier`. It leverages the `SemanticComparison` modules for deeper insight into the nature of the divergence. Upon detection, it analyzes the semantic state and proposes corrective actions.
*   **`strategies.py`**: Contains a repertoire of strategies for automated reconciliation:
    *   **Contextual Re-prompting:** Dynamically generating new prompts for an LLM with additional context, explicit constraints, or examples to guide it towards a desired output.
    *   **Output Refinement:** Guiding the LLM to refine an inconsistent output based on a known good state, a desired schema, or specific missing information.
    *   **State Adjustment:** Proposing a minor, controlled adjustment to the `StateManager`'s state if a detected divergence is semantically minor and can be safely reconciled without full re-execution.
    *   **Rollback & Retry:** Initiating a full rollback to a previous valid checkpoint and retrying the agent action with modified parameters or a different strategy.

### 5. `verification/` (Formal Rules & Invariant Checking)

Ensuring adherence to business logic, system invariants, and expected behaviors.

*   **`rule_engine.py`**: Defines and evaluates business rules, invariants, and constraints that system states and agent outputs must consistently adhere to. Rules can be expressed declaratively (e.g., using a DSL or Pydantic validation rules) and are executed against state transitions and agent outputs.
*   **`test_generator.py`**: Automatically generates tests or assertions based on defined `action_schemas` and `rule_engine` policies. These generated tests are automatically run against agent outputs and state transitions, providing continuous validation.
*   **`verifier.py`**: Integrates with the `RuleEngine` and `SemanticComparison` to continuously validate agent outputs and state transitions in real-time. Its primary goal is to 'collapse byzantine failures' into detectable 'crash failures' by immediately flagging any deviation from expected invariants or semantic equivalence upon retry.

### 6. `utils/` (Shared Utilities)

Common functionalities used across the framework.

*   **`llm_connector.py`**: A standardized, unified interface for interacting with various LLM providers (e.g., OpenAI, Anthropic, local models). It encapsulates essential features like robust retry logic, rate limiting, comprehensive token counting, and potentially output caching or versioning to aid debugging and analysis.
*   **`logging_config.py`**: Centralized logging configuration to ensure consistent, detailed, and actionable logging across all components of the framework.
*   **`serialization.py`**: Utility functions for reliably serializing and deserializing complex objects, which is critical for state checkpointing, persistence, and inter-process communication.

### 7. `config.py`

A centralized configuration file for all framework settings, including LLM API keys, preferred model names, semantic similarity thresholds, logging levels, and state storage backend details.

## Workflow Example: Handling Semantic Inconsistency

Let's illustrate how these components interact in a typical scenario involving an agent and a potential LLM output inconsistency:

1.  **Initialization:** `main.py` initializes the `WorkflowManager`, registers available agents, and configures the system.
2.  **Agent Action:** An `ExampleAgent` is tasked with processing some data, which involves making an LLM call via `utils/llm_connector.py`.
3.  **Proposed State:** The agent proposes its action and the expected output to the `WorkflowManager`.
4.  **State Management & Checkpoint:** The `WorkflowManager` consults the `StateManager` to record the pending state, potentially triggering a state checkpoint to enable future rollbacks.
5.  **Validation - Semantic Comparison (Retry Scenario):** If this agent action is a retry (e.g., the previous attempt failed or timed out), the `WorkflowManager` (or `Verifier`) uses `semantic_comparison/comparators.py` to check if the new LLM output is semantically equivalent to the previous attempt or a predefined expected outcome.
6.  **Validation - Invariant Checking:** Concurrently, the `Verifier` actively uses the `rule_engine.py` and `test_generator.py` to validate the agent's proposed output and the resulting state against all predefined business rules and invariants.
7.  **Inconsistency Detected:** If an inconsistency (semantic divergence or rule violation) is detected by either the `SemanticComparison` modules or the `Verifier`, the `WorkflowManager` is immediately notified.
8.  **Reconciliation Triggered:** The `WorkflowManager` dispatches the detected inconsistency to the `ReconciliationAgent`.
9.  **Proposing Correction:** The `ReconciliationAgent` analyzes the nature of the inconsistency (using `semantic_comparison` for deeper insight), consults its `strategies.py`, and proposes a corrective action (e.g., re-prompting the LLM with refined context, rolling back to a previous consistent state, or applying a minor, controlled state adjustment).
10. **Executing Reconciliation:** The `WorkflowManager` executes the proposed reconciliation strategy, which might involve instructing the `ExampleAgent` to run again with modified parameters or directly updating the `StateManager` under strict controls.

This framework shifts the paradigm from manually debugging opaque byzantine failures to systematic detection, automated analysis, and intelligent reconciliation of semantic inconsistencies, paving the way for truly robust and reliable multi-agent AI systems.