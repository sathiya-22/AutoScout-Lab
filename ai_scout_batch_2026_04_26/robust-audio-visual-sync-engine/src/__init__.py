"""
Robust Audio-Visual Synchronization Engine (RAVSE)
==================================================

This package provides a robust engine for extracting and aligning
temporal cues, such as pitch shifts, from 'in-the-wild' videos.
It leverages advanced audio processing (source separation, pitch tracking)
and audio-visual alignment techniques, providing confidence scores
for extracted cues to ensure downstream robustness for self-supervised
video temporal reasoning.

Key Features:
- Advanced Audio Source Separation for noisy environments.
- State-of-the-art Pitch Tracking.
- Cross-modal Attention and Correlation-based Audio-Visual Alignment.
- Granular Confidence Scoring for all extracted temporal cues.
- Metadata output for audio quality assessment.
- High-level API for batch and real-time processing.
"""

__version__ = "0.1.0"

# Expose the main API class for easy access.
# It's assumed that src/api/sync_api.py defines a class named SyncEngineAPI
# which is the primary interface for users of this package.
SyncEngineAPI = None
try:
    from .api.sync_api import SyncEngineAPI
except ImportError as e:
    print(f"WARNING: Could not import SyncEngineAPI from src.api.sync_api. "
          f"The package may not function correctly. Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred while importing SyncEngineAPI: {e}")


# Optionally, expose the core orchestration engine for more advanced use cases
# or internal debugging/customization.
# It's assumed that src/engine.py defines a class named SyncEngine.
SyncEngine = None
try:
    from .engine import SyncEngine
except ImportError as e:
    print(f"WARNING: Could not import SyncEngine from src.engine. "
          f"Advanced functionalities requiring the core engine might be unavailable. Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred while importing SyncEngine: {e}")


# Define what is exposed when 'from src import *' is used.
__all__ = ["__version__", "SyncEngineAPI", "SyncEngine"]

# Any package-level initialization can go here.
# For example, setting up default logging or checking for critical dependencies.
# For now, we'll keep it minimal.