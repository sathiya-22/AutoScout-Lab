"""
Orchestration and Coordination Framework for Multi-Agent AI Systems.

This package provides the foundational components for building robust, scalable,
and cost-effective multi-agent AI systems, focusing on communication,
task management, resource allocation, and overall system coherence.

Modules:
- `core`: Contains the central orchestration logic, agent abstraction,
  task, and resource management.
- `communication`: Provides primitives for inter-agent communication,
  including message queues, shared memory (stigmergy), and RPC.
- `agents`: Defines base agent classes and provides examples of specialized
  LLM-powered agents.
- `monitoring`: Offers tools for metrics, logging, and distributed tracing
  to ensure observability.
- `config`: Manages system-wide configurations.
- `examples`: Demonstrates various orchestration patterns and framework usage.
"""

__version__ = "0.1.0"

# This __init__.py primarily serves to mark the `src` directory as a Python package
# and expose a version. Specific components are expected to be imported directly
# from their respective sub-modules (e.g., `from src.core.orchestrator import Orchestrator`).
# No direct imports are made here to maintain modularity and avoid potential
# circular dependencies or namespace pollution at the top level.