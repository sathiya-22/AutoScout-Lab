from . import source_separation
from . import pitch_tracking
from . import feature_extraction

# Expose core classes/functions from submodules for easier access
# Assuming the main class in each submodule is named similarly to the submodule itself
# e.g., SourceSeparator in source_separation.py
try:
    from .source_separation import SourceSeparator
except ImportError:
    SourceSeparator = None
    print("Warning: Could not import SourceSeparator. Source separation functionality may be unavailable.")

try:
    from .pitch_tracking import PitchTracker
except ImportError:
    PitchTracker = None
    print("Warning: Could not import PitchTracker. Pitch tracking functionality may be unavailable.")

try:
    from .feature_extraction import AudioFeatureExtractor
except ImportError:
    AudioFeatureExtractor = None
    print("Warning: Could not import AudioFeatureExtractor. Feature extraction functionality may be unavailable.")


__all__ = [
    "SourceSeparator",
    "PitchTracker",
    "AudioFeatureExtractor",
    "source_separation", # Make the submodule itself accessible if needed for specific functions
    "pitch_tracking",
    "feature_extraction",
]