```python
import numpy as np
import logging
import librosa
from typing import Tuple, Literal

# --- Conditional Imports for Pitch Trackers ---
_has_crepe = False
try:
    import crepe
    _has_crepe = True
except ImportError:
    pass # Logged during PitchTracker instantiation if selected

_has_pyworld = False
try:
    import pyworld
    _has_pyworld = True
except ImportError:
    pass # Logged during PitchTracker instantiation if selected

# --- Setup Logging ---
logger = logging.getLogger(__name__)
# Configure basic logging if not already configured by a parent application
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PitchTracker:
    """
    Applies state-of-the-art pitch tracking algorithms to audio streams.

    This class supports CREPE and WORLD algorithms, designed to work with
    already isolated audio streams (e.g., from source separation) for
    cleaner and more accurate pitch extraction.
    """

    # Default parameters, can be overridden in __init__
    DEFAULT_SR = 16000
    DEFAULT_HOP_LENGTH_MS = 10  # milliseconds for frame period/step size
    DEFAULT_CREPE_MODEL = 'tiny' # 'tiny', 'full', 'medium', 'large', 'resnet'
    DEFAULT_TRACKER_TYPE = 'crepe'
    # Confidence threshold below which a pitch estimate is considered unvoiced (and set to 0 Hz)
    UNVOICED_CONFIDENCE_THRESHOLD = 0.1

    def __init__(self,
                 sr: int = DEFAULT_SR,
                 hop_length_ms: int = DEFAULT_HOP_LENGTH_MS,
                 tracker_type: Literal['crepe', 'world'] = DEFAULT_TRACKER_TYPE,
                 crepe_model: Literal['tiny', 'full', 'medium', 'large', 'resnet'] = DEFAULT_CREPE_MODEL):
        """
        Initializes the PitchTracker with a specific algorithm and parameters.

        Args:
            sr (int): Sample rate of the audio to be processed. Input audio will be
                      resampled to this rate if necessary.
            hop_length_ms (int): The hop length in milliseconds for pitch estimation.
                                 This determines the temporal resolution of the pitch contour.
            tracker_type (Literal['crepe', 'world']): The pitch tracking algorithm to use.
            crepe_model (Literal['tiny', 'full', 'medium', 'large', 'resnet']):
                         Model size for CREPE (ignored if tracker_type is 'world').
        """
        if not isinstance(sr, int) or sr <= 0:
            raise ValueError(f"Invalid sample rate: {sr}. Must be a positive integer.")
        if not isinstance(hop_length_ms, (int, float)) or hop_length_ms <= 0:
            raise ValueError(f"Invalid hop_length_ms: {hop_length_ms}. Must be a positive number.")
        if tracker_type not in ['crepe', 'world']:
            raise ValueError(f"Invalid tracker_type: {tracker_type}. Must be 'crepe' or 'world'.")
        if crepe_model not in ['tiny', 'full', 'medium', 'large', 'resnet']:
            raise ValueError(f"Invalid crepe_model: {crepe_model}. Must be one of 'tiny', 'full', 'medium', 'large', 'resnet'.")

        if tracker_type == 'crepe' and not _has_crepe:
            raise ImportError(
                f"CREPE is selected but not installed. Please install it with 'pip install crepe'. "
                "Alternatively, choose 'world' as tracker_type."
            )
        if tracker_type == 'world' and not _has_pyworld:
            raise ImportError(
                f"PyWORLD is selected but not installed. Please install it with 'pip install pyworld'. "
                "Alternatively, choose 'crepe' as tracker_type."
            )

        self.sr = sr
        self.hop_length_ms = hop_length_ms
        self.tracker_type = tracker_type
        self.crepe_model = crepe_model

        logger.info(f"Initialized PitchTracker: type='{self.tracker_type}', sr={self.sr} Hz, hop_length_ms={self.hop_length_ms} ms")
        if self.tracker_type == 'crepe':
            logger.info(f"  CREPE model size: {self.crepe_model}")

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocesses the audio signal for pitch tracking.
        Ensures float32 type and normalizes to -1.0 to 1.0 if not already.

        Args:
            audio (np.ndarray): Input audio signal.

        Returns:
            np.ndarray: Preprocessed mono audio signal.
        """
        if not isinstance(audio, np.ndarray):
            raise TypeError("Audio input must be a numpy array.")

        # Ensure audio is mono; if stereo, take the first channel
        if audio.ndim > 1:
            if audio.shape[0] == 2 and audio.shape[1] > 2: # Librosa loads as (channels, samples)
                logger.warning("Input audio is multi-channel. Taking the first channel.")
                audio = audio[0]
            elif audio.shape[1] == 2 and audio.shape[0] > 2: # Scipy loads as (samples, channels)
                logger.warning("Input audio is multi-channel. Taking the first channel.")
                audio = audio[:, 0]
            else: # Ambiguous case, assume it's already mono or take first
                logger.warning(f"Input audio shape {audio.shape} is ambiguous. Attempting to convert to mono by taking first channel/row.")
                audio = audio.flatten() # Fallback, might not be ideal

        # Ensure float32 for most audio processing libraries
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize to -1 to 1 to prevent issues with some models, especially deep learning ones.
        # This is a basic normalization; assuming audio is generally within a reasonable range.
        # For already separated tracks, this might be less critical if source separation
        # handles normalization, but it's a safe guard.
        peak = np.max(np.abs(audio))
        if peak > 0: # Avoid division by zero for silent tracks
            if peak > 1.0:
                audio = audio / peak
                logger.debug(f"Normalized audio peak from {peak:.2f} to 1.0.")
        else:
            logger.warning("Input audio is all zeros (silent). Pitch tracking will likely yield no pitch.")

        return audio

    def _track_pitch_crepe(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tracks pitch using the CREPE algorithm.

        Args:
            audio (np.ndarray): Preprocessed mono audio signal.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - times (np.ndarray): Timestamps of pitch estimates in seconds.
                - frequencies (np.ndarray): Estimated fundamental frequencies (Hz).
                - confidences (np.ndarray): Confidence scores for each estimate (0.0 to 1.0).
        """
        if not _has_crepe:
            raise RuntimeError("CREPE library is not available. Check installation.")

        try:
            # crepe.predict returns (time, frequency, confidence, activation)
            times, frequencies, confidences, _ = crepe.predict(
                audio,
                self.sr,
                viterbi=True, # Use Viterbi decoding for smoother pitch contours
                stepsize=self.hop_length_ms, # hop_length in milliseconds
                model=self.crepe_model,
                verbose=False # Suppress CREPE's internal progress bar
            )
            logger.debug(f"CREPE tracking complete. Found {len(frequencies)} pitch frames.")
            return times, frequencies, confidences
        except Exception as e:
            logger.error(f"Error during CREPE pitch tracking: {e}")
            raise RuntimeError(f"CREPE pitch tracking failed: {e}")

    def _track_pitch_world(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tracks pitch using the WORLD vocoder's F0 estimation (Harvest algorithm).

        Args:
            audio (np.ndarray): Preprocessed mono audio signal.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - times (np.ndarray): Timestamps of pitch estimates in seconds.
                - frequencies (np.ndarray): Estimated fundamental frequencies (Hz).
                - confidences (np.ndarray): Confidence scores (1.0 for voiced, 0.0 for unvoiced).
        """
        if not _has_pyworld:
            raise RuntimeError("PyWORLD library is not available. Check installation.")

        try:
            # Harvest is generally more robust for noisy signals than DIO
            # frame_period is in milliseconds
            f0, t = pyworld.harvest(
                audio,
                self.sr,
                frame_period=self.hop_length_ms
            )

            # WORLD does not provide a direct 'confidence' like CREPE.
            # We approximate by marking non-zero F0 values as 'voiced' (confidence 1)
            # and zero F0 values (unvoiced) as 'confidence 0).
            confidences = np.where(f0 > 0, 1.0, 0.0)

            logger.debug(f"WORLD tracking complete. Found {len(f0)} pitch frames.")
            return t, f0, confidences
        except Exception as e:
            logger.error(f"Error during WORLD pitch tracking: {e}")
            raise RuntimeError(f"WORLD pitch tracking failed: {e}")

    def track_pitch(self, audio: np.ndarray, sr: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main method to track pitch for a given audio segment.

        Args:
            audio (np.ndarray): Mono audio signal (expected to be float type).
            sr (int, optional): Sample rate of the input audio. If None, uses the
                                 sample rate defined during PitchTracker initialization.
                                 If different, the audio will be resampled to `self.sr`.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - times (np.ndarray): Timestamps (in seconds) corresponding to each pitch estimate.
                - frequencies (np.ndarray): Estimated fundamental frequencies (F0) in Hz.
                                            Unvoiced frames will have F0 ~0 Hz.
                - confidences (np.ndarray): Confidence scores for each pitch estimate (0.0 to 1.0).

        Raises:
            ValueError: If input audio is invalid.
            ImportError: If the chosen pitch tracker library or librosa (for resampling) is not installed.
            RuntimeError: If an error occurs during pitch tracking.
        """
        if audio is None or audio.size == 0:
            logger.warning("Received empty or None audio input for pitch tracking.")
            return np.array([]), np.array([]), np.array([])

        current_sr = sr if sr is not None else self.sr

        # Resample if input SR differs from tracker's configured SR
        if current_sr != self.sr:
            logger.info(f"Input audio sample rate ({current_sr} Hz) differs from tracker's configured rate ({self.sr} Hz). "
                        f"Resampling audio from {current_sr} Hz to {self.sr} Hz.")
            try:
                # Ensure librosa is imported for resampling
                if not hasattr(librosa, 'resample'):
                    raise ImportError("librosa is required for audio resampling. Please install it with 'pip install librosa'.")
                audio = librosa.resample(y=audio.astype(np.float32), orig_sr=current_sr, target_sr=self.sr)
            except Exception as e:
                logger.error(f"Failed to resample audio from {current_sr} Hz to {self.sr} Hz: {e}")
                raise RuntimeError(f"Audio resampling failed: {e}")
        
        processed_audio = self._preprocess_audio(audio)

        if self.tracker_type == 'crepe':
            times, frequencies, confidences = self._track_pitch_crepe(processed_audio)
        elif self.tracker_type == 'world':
            times, frequencies, confidences = self._track_pitch_world(processed_audio)
        else:
            # This case should ideally be caught in __init__, but as a safeguard:
            raise RuntimeError(f"Unknown pitch tracker type: {self.tracker_type}")

        # Post-process frequencies: replace low F0s (often noise or tracking errors for unvoiced) with 0.
        # This standardizes output where 0 Hz means unvoiced or undetected pitch.
        frequencies[confidences < self.UNVOICED_CONFIDENCE_THRESHOLD] = 0.0

        return times, frequencies, confidences
```