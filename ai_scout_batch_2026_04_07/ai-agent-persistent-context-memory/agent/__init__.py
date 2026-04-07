import sys

try:
    from . import core_agent
    from . import memory_subsystem
    from . import checkpoint_manager
    from . import knowledge_crystallizer
    from . import context_optimizer
    from . import state_manager # Manages agent's current state, used by core_agent

    # Optionally, expose key classes/functions directly for easier access
    # from .core_agent import CoreAgent
    # from .memory_subsystem import MemorySubsystem
    # from .checkpoint_manager import CheckpointManager
    # from .knowledge_crystallizer import KnowledgeCrystallizer
    # from .context_optimizer import ContextOptimizer
    # from .state_manager import StateManager

    __all__ = [
        "core_agent",
        "memory_subsystem",
        "checkpoint_manager",
        "knowledge_crystallizer",
        "context_optimizer",
        "state_manager",
        # "CoreAgent", # if directly exposed
    ]

except ImportError as e:
    # Log the error or handle it as appropriate.
    # For a core package like 'agent', a failure to import its essential components
    # usually indicates a critical setup issue (e.g., missing files, dependency problems).
    # Re-raising the error is typically the most appropriate action in __init__.py
    # to prevent the application from starting in a broken state.
    print(f"CRITICAL ERROR: Failed to initialize the 'agent' package. Missing components or dependencies.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    raise # Re-raise to halt execution or propagate the error up

__version__ = "0.1.0" # Define a version for the agent package