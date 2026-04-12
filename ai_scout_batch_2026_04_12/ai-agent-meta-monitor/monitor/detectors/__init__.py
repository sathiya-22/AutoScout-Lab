from .base_detector import BaseDetector, ProblemReport
from .loop_detector import LoopDetector
from .progress_detector import ProgressDetector
from .critique_agent_detector import CritiqueAgentDetector

# A convenience list to easily get all concrete detector implementations
# for registration in the Monitor Core.
ALL_DETECTOR_CLASSES = [
    LoopDetector,
    ProgressDetector,
    CritiqueAgentDetector,
]

__all__ = [
    "BaseDetector",
    "ProblemReport",
    "LoopDetector",
    "ProgressDetector",
    "CritiqueAgentDetector",
    "ALL_DETECTOR_CLASSES",
]