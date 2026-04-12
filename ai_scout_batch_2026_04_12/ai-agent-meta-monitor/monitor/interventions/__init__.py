from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

# Conditional import for type checking to prevent circular dependencies
if TYPE_CHECKING:
    from monitor.detectors.base_detector import ProblemReport
    from agents.base_agent import BaseAgent # Assuming BaseAgent defines the agent interface

class BaseIntervener(ABC):
    """
    Abstract base class for all intervention strategies.
    Interveners are responsible for taking corrective action when a detector
    signals a problem with the monitored agent.
    """

    def __init__(self, name: str):
        """
        Initializes the base intervener.

        Args:
            name: A unique name for this intervener.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Intervener name must be a non-empty string.")
        self._name = name

    @property
    def name(self) -> str:
        """Returns the name of the intervener."""
        return self._name

    @abstractmethod
    def intervene(self, agent: "BaseAgent", problem_report: "ProblemReport") -> bool:
        """
        Applies an intervention to the agent based on the reported problem.

        Args:
            agent: The instance of the agent to intervene upon. This agent
                   is expected to implement methods to accept interventions
                   (e.g., replan, receive_hint).
            problem_report: An object detailing the detected problem,
                            including its type and context.

        Returns:
            True if the intervention was successfully applied, False otherwise.
            This allows the Monitor Core to handle cases where an intervention
            might fail or be rejected by the agent.
        """
        pass

# Import concrete interveners to make them part of the 'interventions' package.
# This makes them easily discoverable and importable from monitor.interventions.*
try:
    from .replan_intervener import ReplanIntervener
    from .hint_intervener import HintIntervener
    from .human_fallback import HumanFallbackIntervener
except ImportError as e:
    # Log or handle the error if a specific intervener cannot be imported.
    # For a simple __init__.py, re-raising might be acceptable if the file is critical.
    # In a more robust system, this might log and skip loading the problematic intervener.
    print(f"Warning: Could not import an intervener module: {e}")
    # Depending on requirements, you might want to suppress these imports if the component
    # is optional, or ensure they are critical. For this prototype, we'll re-raise
    # if it's a critical component. For a production system, more graceful handling
    # (e.g., making specific interveners optional) might be preferred.
    pass

# Optionally, define __all__ to explicitly control what gets imported when
# 'from monitor.interventions import *' is used.
__all__ = [
    "BaseIntervener",
    "ReplanIntervener",
    "HintIntervener",
    "HumanFallbackIntervener",
]