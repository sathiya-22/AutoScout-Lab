from .context_injector import ContextInjector
from .state_manager import StateManager
from .data_referencer import DataReferencer

__all__ = [
    "ContextInjector",
    "StateManager",
    "DataReferencer",
]

# This __init__.py makes the 'anchoring' directory a Python package.
# It also provides convenient access to the main components of the anchoring layer
# by importing them directly into the package's namespace.
#
# The anchoring layer is responsible for injecting crucial context into LLM prompts
# and managing internal states to reduce variability and enhance reproducibility.