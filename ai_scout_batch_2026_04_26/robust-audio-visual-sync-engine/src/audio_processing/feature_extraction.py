```python
import numpy as np
import librosa
import logging

# Set up a basic logger for error reporting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureExtractionError(Exception):
    """Custom exception for feature extraction errors."""
    pass

def _validate_audio_input(y: np.ndarray, sr: int):
    """Internal helper to validate audio input."""
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.size == 0:
        raise FeatureExtractionError("Invalid audio waveform: 'y' must be a non-empty 1D numpy array.")
    if not isinstance(sr, int) or sr <= 0:
        raise FeatureExtractionError("Invalid sample rate: 'sr' must be a positive integer.")

def extract_spectrogram(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512,
                        return_log_spectrogram: bool = True, **kwargs) -> np.ndarray:
    """
    Extracts the magnitude spectrogram from an audio waveform.

    Args:
        y (np.ndarray): Audio time series (expected to be float type).
        sr (int): Sampling rate of `y`.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of samples between successive frames.
        return_log_spectrogram (bool): If True, returns log-amplitude spectrogram (dB),
                                       otherwise returns magnitude spectrogram.
        **kwargs: Additional keyword arguments for librosa.stft.

    Returns:
        np.ndarray: The magnitude or log-magnitude spectrogram.

    Raises:
        FeatureExtractionError: If input validation fails or librosa encounters an error.
    """
    try:
        _validate_audio_input(y, sr)
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, **kwargs))

        if return_log_spectrogram:
            # Convert to decibels, using a small epsilon to avoid log(0)
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            return S_db
        else:
            return S
    except FeatureExtractionError:
        raise # Re-raise custom errors directly
    except Exception as e:
        logging.error(f"Error extracting spectrogram: {e}", exc_info=True)
        raise FeatureExtractionError(f"Failed to extract spectrogram: {e}")

def extract_mel_spectrogram(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512,
                            n_mels: int = 128, fmax: float = None, return_log_mel: bool = True,
                            **kwargs) -> np.ndarray:
    """
    Extracts the Mel-scaled spectrogram from an audio waveform.

    Args:
        y (np.ndarray): Audio time series (expected to be float type).
        sr (int): Sampling rate of `y`.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of samples between successive frames.
        n_mels (int): Number of Mel bands to generate.
        fmax (float, optional): Maximum frequency (hz). Defaults to sr / 2.
        return_log_mel (bool): If True, returns log-amplitude Mel spectrogram (dB),
                               otherwise returns Mel magnitude spectrogram.
        **kwargs: Additional keyword arguments for librosa.feature.melspectrogram.

    Returns:
        np.ndarray: The Mel spectrogram (magnitude or log-magnitude).

    Raises:
        FeatureExtractionError: If input validation fails or librosa encounters an error.
    """
    try:
        _validate_audio_input(y, sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                  n_mels=n_mels, fmax=fmax, **kwargs)
        if return_log_mel:
            mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        else:
            return mel_spec
    except FeatureExtractionError:
        raise
    except Exception as e:
        logging.error(f"Error extracting Mel spectrogram: {e}", exc_info=True)
        raise FeatureExtractionError(f"Failed to extract Mel spectrogram: {e}")

def extract_mfccs(y: np.ndarray, sr: int, n_mfcc: int = 20, n_fft: int = 2048,
                  hop_length: int = 512, **kwargs) -> np.ndarray:
    """
    Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio waveform.

    Args:
        y (np.ndarray): Audio time series (expected to be float type).
        sr (int): Sampling rate of `y`.
        n_mfcc (int): Number of MFCCs to return.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of samples between successive frames.
        **kwargs: Additional keyword arguments for librosa.feature.mfcc.

    Returns:
        np.ndarray: The MFCC matrix.

    Raises:
        FeatureExtractionError: If input validation fails or librosa encounters an error.
    """
    try:
        _validate_audio_input(y, sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                     hop_length=hop_length, **kwargs)
        return mfccs
    except FeatureExtractionError:
        raise
    except Exception as e:
        logging.error(f"Error extracting MFCCs: {e}", exc_info=True)
        raise FeatureExtractionError(f"Failed to extract MFCCs: {e}")

def extract_loudness_envelope(y: np.ndarray, sr: int, frame_length: int = 2048,
                              hop_length: int = 512, unit: str = 'db', **kwargs) -> np.ndarray:
    """
    Extracts the loudness envelope (RMS energy) from an audio waveform.

    Args:
        y (np.ndarray): Audio time series (expected to be float type).
        sr (int): Sampling rate of `y`.
        frame_length (int): Length of the frame over which to compute RMS.
        hop_length (int): Number of samples between successive frames.
        unit (str): Unit for the loudness envelope. 'db' for decibels, 'amplitude' for raw amplitude.
        **kwargs: Additional keyword arguments for librosa.feature.rms.

    Returns:
        np.ndarray: The loudness envelope.

    Raises:
        FeatureExtractionError: If input validation fails or librosa encounters an error.
    """
    try:
        _validate_audio_input(y, sr)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, **kwargs)[0]

        if unit == 'db':
            # Convert to decibels, using a small epsilon to avoid log(0)
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            return rms_db
        elif unit == 'amplitude':
            return rms
        else:
            raise ValueError("Unit must be 'db' or 'amplitude'.")
    except FeatureExtractionError:
        raise
    except Exception as e:
        logging.error(f"Error extracting loudness envelope: {e}", exc_info=True)
        raise FeatureExtractionError(f"Failed to extract loudness envelope: {e}")

def extract_zero_crossing_rate(y: np.ndarray, sr: int, frame_length: int = 2048,
                               hop_length: int = 512, **kwargs) -> np.ndarray:
    """
    Extracts the zero-crossing rate (ZCR) from an audio waveform.
    ZCR is a measure of the number of times the signal crosses the zero axis.

    Args:
        y (np.ndarray): Audio time series (expected to be float type).
        sr (int): Sampling rate of `y`.
        frame_length (int): Length of the frame over which to compute ZCR.
        hop_length (int): Number of samples between successive frames.
        **kwargs: Additional keyword arguments for librosa.feature.zero_crossing_rate.

    Returns:
        np.ndarray: The zero-crossing rate.

    Raises:
        FeatureExtractionError: If input validation fails or librosa encounters an error.
    """
    try:
        _validate_audio_input(y, sr)
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length, **kwargs)[0]
        return zcr
    except FeatureExtractionError:
        raise
    except Exception as e:
        logging.error(f"Error extracting zero-crossing rate: {e}", exc_info=True)
        raise FeatureExtractionError(f"Failed to extract zero-crossing rate: {e}")


def extract_chroma_stft(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512,
                        n_chroma: int = 12, **kwargs) -> np.ndarray:
    """
    Extracts Chroma features (chromagram) from an audio waveform using STFT.
    Chroma features are typically used to characterize harmonic content.

    Args:
        y (np.ndarray): Audio time series (expected to be float type).
        sr (int): Sampling rate of `y`.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of samples between successive frames.
        n_chroma (int): Number of chroma bins to produce.
        **kwargs: Additional keyword arguments for librosa.feature.chroma_stft.

    Returns:
        np.ndarray: The Chroma STFT matrix.

    Raises:
        FeatureExtractionError: If input validation fails or librosa encounters an error.
    """
    try:
        _validate_audio_input(y, sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                            n_chroma=n_chroma, **kwargs)
        return chroma
    except FeatureExtractionError:
        raise
    except Exception as e:
        logging.error(f"Error extracting Chroma STFT: {e}", exc_info=True)
        raise FeatureExtractionError(f"Failed to extract Chroma STFT: {e}")

def extract_all_features(y: np.ndarray, sr: int, **kwargs) -> dict:
    """
    Extracts a comprehensive set of common audio features from an audio waveform.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.
        **kwargs: Optional keyword arguments for individual feature extraction functions.
                  E.g., `mfcc_n_mfcc=40`, `spectrogram_return_log=False`.
                  Parameters are passed as `featurename_paramname`.

    Returns:
        dict: A dictionary where keys are feature names and values are the extracted feature matrices.
              If a feature extraction fails, its value will be None in the dictionary.

    Raises:
        FeatureExtractionError: If input validation fails or *all* feature extractions encounter an error.
    """
    _validate_audio_input(y, sr)
    
    # Convert to float once for all feature extractions for consistency and efficiency
    y_float = librosa.to_float(y) 

    features = {}
    total_failures = 0
    total_features_attempted = 0

    # Extract Spectrogram (log-magnitude by default)
    total_features_attempted += 1
    try:
        spec_kwargs = {k.replace('spectrogram_', ''): v for k, v in kwargs.items() if k.startswith('spectrogram_')}
        features['spectrogram'] = extract_spectrogram(y_float, sr, **spec_kwargs)
    except FeatureExtractionError as e:
        logging.warning(f"Skipping spectrogram extraction due to error: {e}")
        features['spectrogram'] = None
        total_failures += 1

    # Extract Mel Spectrogram (log-magnitude by default)
    total_features_attempted += 1
    try:
        mel_spec_kwargs = {k.replace('mel_spectrogram_', ''): v for k, v in kwargs.items() if k.startswith('mel_spectrogram_')}
        features['mel_spectrogram'] = extract_mel_spectrogram(y_float, sr, **mel_spec_kwargs)
    except FeatureExtractionError as e:
        logging.warning(f"Skipping Mel spectrogram extraction due to error: {e}")
        features['mel_spectrogram'] = None
        total_failures += 1

    # Extract MFCCs
    total_features_attempted += 1
    try:
        mfcc_kwargs = {k.replace('mfcc_', ''): v for k, v in kwargs.items() if k.startswith('mfcc_')}
        features['mfccs'] = extract_mfccs(y_float, sr, **mfcc_kwargs)
    except FeatureExtractionError as e:
        logging.warning(f"Skipping MFCCs extraction due to error: {e}")
        features['mfccs'] = None
        total_failures += 1

    # Extract Loudness Envelope (RMS in dB by default)
    total_features_attempted += 1
    try:
        loudness_kwargs = {k.replace('loudness_envelope_', ''): v for k, v in kwargs.items() if k.startswith('loudness_envelope_')}
        features['loudness_envelope'] = extract_loudness_envelope(y_float, sr, **loudness_kwargs)
    except FeatureExtractionError as e:
        logging.warning(f"Skipping loudness envelope extraction due to error: {e}")
        features['loudness_envelope'] = None
        total_failures += 1

    # Extract Zero-Crossing Rate
    total_features_attempted += 1
    try:
        zcr_kwargs = {k.replace('zcr_', ''): v for k, v in kwargs.items() if k.startswith('zcr_')}
        features['zero_crossing_rate'] = extract_zero_crossing_rate(y_float, sr, **zcr_kwargs)
    except FeatureExtractionError as e:
        logging.warning(f"Skipping zero-crossing rate extraction due to error: {e}")
        features['zero_crossing_rate'] = None
        total_failures += 1

    # Extract Chroma STFT
    total_features_attempted += 1
    try:
        chroma_kwargs = {k.replace('chroma_stft_', ''): v for k, v in kwargs.items() if k.startswith('chroma_stft_')}
        features['chroma_stft'] = extract_chroma_stft(y_float, sr, **chroma_kwargs)
    except FeatureExtractionError as e:
        logging.warning(f"Skipping Chroma STFT extraction due to error: {e}")
        features['chroma_stft'] = None
        total_failures += 1

    # If all attempted features failed, raise a general error
    if total_features_attempted > 0 and total_failures == total_features_attempted:
        raise FeatureExtractionError("All attempted feature extractions failed for the given audio input.")

    return features
```