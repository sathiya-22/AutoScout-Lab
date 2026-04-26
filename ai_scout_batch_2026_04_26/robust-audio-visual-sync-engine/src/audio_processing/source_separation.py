```python
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Union, List, Tuple
import torch
import torchaudio
import tempfile

# We'll use Demucs as the primary example for source separation.
# The following block attempts to import necessary Demucs components.
# If Demucs or its dependencies (like torch, torchaudio) are not installed,
# a warning will be printed and DEMUCS_AVAILABLE will be set to False,
# disabling source separation functionality to prevent crashes.
try:
    from demucs.api import Separator
    from demucs.utils import load_audio
    DEMUCS_AVAILABLE = True
except ImportError:
    print("Warning: Demucs library or its dependencies (e.g., torch, torchaudio) not found. "
          "Source separation functionality will be disabled. "
          "Please install Demucs with `pip install demucs`.")
    DEMUCS_AVAILABLE = False
except Exception as e:
    print(f"Warning: Failed to import Demucs components due to an unexpected error: {e}. "
          "Source separation functionality will be disabled.")
    DEMUCS_AVAILABLE = False


class SourceSeparator:
    """
    A class for performing audio source separation using deep learning models.
    Currently supports Demucs for separating common audio sources like vocals, drums, bass, etc.
    This module is crucial for robust audio processing in 'in-the-wild' videos,
    as it aims to isolate target audio streams from background noise,
    providing cleaner signals for downstream analysis like pitch tracking.
    """

    def __init__(self, model_name: str = "htdemucs_6s", device: str = "cpu", streams: List[str] = None):
        """
        Initializes the SourceSeparator.

        Args:
            model_name (str): The name of the Demucs model to use (e.g., "htdemucs_6s", "htdemucs").
                              Refer to Demucs documentation for available models.
            device (str): The device to run the model on ('cuda' for GPU, 'cpu' for CPU).
                          If 'cuda' is specified but no GPU is available, it will gracefully fall back to 'cpu'.
            streams (List[str], optional): A list of specific streams to extract (e.g., ["vocals", "bass"]).
                                           If None, all available streams for the chosen model will be extracted.
        
        Raises:
            RuntimeError: If Demucs library is not available.
        """
        if not DEMUCS_AVAILABLE:
            raise RuntimeError("Demucs library is not available. Please install it (`pip install demucs`) to use SourceSeparator.")

        self.model_name = model_name
        self.device = device
        self.streams = streams
        self._model: Separator = None  # Demucs model will be lazy-loaded on first separation call.

        # Handle device selection and provide user feedback
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA device requested but not available. Falling back to CPU for Demucs processing.")
            self.device = "cpu"
        elif self.device == "cuda" and torch.cuda.is_available():
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)} for Demucs processing.")
        else:
            print("Using CPU device for Demucs processing.")

        # Print a general initialization message; specific model loading happens lazily
        print(f"SourceSeparator initialized with model configuration (model: '{self.model_name}'). "
              "The Demucs model will be loaded into memory on its first use.")

    def _load_demucs_model(self):
        """
        Lazy loads the Demucs model into memory. This method is called internally
        before the first actual separation task to optimize resource usage.
        """
        if self._model is None:
            print(f"Attempting to load Demucs model '{self.model_name}' on device '{self.device}'...")
            try:
                # The Separator class handles downloading the model weights if they don't exist locally.
                self._model = Separator(model=self.model_name, device=self.device)
                print("Demucs model loaded successfully.")
            except Exception as e:
                self._model = None # Ensure the model is None if loading fails
                raise RuntimeError(f"Failed to load Demucs model '{self.model_name}' on device '{self.device}': {e}. "
                                   "This could be due to a missing model file (Demucs downloads on first use), "
                                   "an incorrect model name, or device-related issues.")

    def _separate_with_demucs(self, audio_path: Path) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Internal method to perform source separation using the Demucs library on an audio file.

        Args:
            audio_path (Path): Path to the input audio file.

        Returns:
            Tuple[Dict[str, np.ndarray], int]: A dictionary where keys are stream names
                                            (e.g., 'vocals', 'drums') and values are the
                                            separated audio data as numpy arrays (mono).
                                            The second element is the sample rate of the output tracks.

        Raises:
            RuntimeError: If the Demucs model fails to load or separation encounters an error.
        """
        self._load_demucs_model() # Ensure the model is loaded before processing

        try:
            # Demucs's `load_audio` handles loading an audio file and automatically
            # resampling it to the model's expected sample rate, which is efficient.
            # The original sample rate is ignored here as Demucs will resample.
            mix_audio_tensor, _ = load_audio(str(audio_path), self._model.samplerate)
            
            # Demucs's `separate_tensor` method expects a batch dimension (B, C, T),
            # where B is batch size, C is channels, T is time (samples).
            # If the input is (C, T), we add a batch dimension.
            if mix_audio_tensor.ndim == 2: # Format: (channels, samples)
                mix_audio_tensor = mix_audio_tensor.unsqueeze(0) # Convert to (1, channels, samples)
            
            # Move the input tensor to the specified processing device (CPU or CUDA GPU)
            mix_audio_tensor = mix_audio_tensor.to(self.device)

            print(f"Starting source separation for '{audio_path.name}' using '{self.model_name}' on '{self.device}'...")
            # Perform the separation. `separated_waveforms` is an OrderedDict where keys are source names
            # and values are torch.Tensor objects, typically (channels, samples) for each source.
            separated_waveforms = self._model.separate_tensor(mix_audio_tensor)
            print(f"Source separation for '{audio_path.name}' completed.")

            separated_tracks = {}
            for source, audio_tensor in separated_waveforms.items():
                # For most downstream audio analysis tasks (like pitch tracking), mono audio is preferred.
                # If the separated track is stereo (multiple channels), average them to create a mono signal.
                if audio_tensor.ndim == 2 and audio_tensor.shape[0] > 1: # Check if stereo (channels, samples)
                    mono_audio = audio_tensor.mean(dim=0)
                else: # Already mono or single channel, ensure it's a 1D tensor (samples,)
                    mono_audio = audio_tensor.squeeze() # Removes any singleton dimensions (e.g., from (1, samples) to (samples,))

                # Convert the PyTorch tensor to a NumPy array and move it to CPU memory
                separated_tracks[source] = mono_audio.cpu().numpy()

            return separated_tracks, self._model.samplerate

        except Exception as e:
            # General exception catch for Demucs processing issues
            print(f"Error during Demucs separation for '{audio_path.name}': {e}")
            raise RuntimeError(f"Demucs separation failed for '{audio_path.name}': {e}")

    def separate_audio(self, audio_input: Union[str, Path, np.ndarray], sample_rate: int = None) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Performs source separation on the given audio input, which can be a file path or a NumPy array.

        Args:
            audio_input (Union[str, Path, np.ndarray]): Path to the audio file (string or Path object)
                                                        or a NumPy array representing raw audio data.
            sample_rate (int, optional): The sample rate of the input audio data. This parameter is
                                         REQUIRED if `audio_input` is a NumPy array. It is ignored
                                         if `audio_input` is a file path, as the sample rate will
                                         be read from the file itself.

        Returns:
            Tuple[Dict[str, np.ndarray], int]: A dictionary where keys are stream names (e.g., 'vocals', 'drums')
                                            and values are the separated audio data as NumPy arrays (mono).
                                            The second element is the sample rate of the output tracks (which
                                            is determined by the Demucs model's internal sample rate).
                                            Returns an empty dictionary and 0 if separation fails or inputs are invalid.

        Raises:
            ValueError: If `audio_input` is a NumPy array but `sample_rate` is not provided or is invalid.
            FileNotFoundError: If the provided audio file path does not exist.
            TypeError: If the `audio_input` type is not supported (neither string, Path, nor NumPy array).
            RuntimeError: If Demucs is not available or if there's an issue during the separation process.
        """
        if not DEMUCS_AVAILABLE:
            raise RuntimeError("Demucs library is not available. Source separation cannot be performed.")

        temp_file_obj = None  # Used to manage temporary file creation and cleanup
        output_sr = 0         # Initialize output sample rate

        try:
            audio_path: Path
            if isinstance(audio_input, (str, Path)):
                # If input is a file path, validate it
                audio_path = Path(audio_input)
                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: '{audio_path}'")
                if audio_path.stat().st_size == 0:
                    print(f"Warning: Input audio file '{audio_path}' is empty. Returning empty separation results.")
                    return {}, 0
            elif isinstance(audio_input, np.ndarray):
                # If input is a NumPy array, validate sample_rate and array content
                if sample_rate is None:
                    raise ValueError("`sample_rate` must be provided when `audio_input` is a numpy array.")
                if not isinstance(sample_rate, int) or sample_rate <= 0:
                    raise ValueError(f"Invalid `sample_rate`: {sample_rate}. Must be a positive integer.")
                if audio_input.size == 0:
                    print("Warning: Input audio array is empty. Returning empty separation results.")
                    return {}, 0
                
                # Demucs's `load_audio` expects a file path. Therefore, write the NumPy array to a temporary WAV file.
                # `tempfile.NamedTemporaryFile` is used for robust temporary file management.
                temp_file_obj = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                audio_path = Path(temp_file_obj.name)
                temp_file_obj.close() # Close the file handle; `soundfile.write` will open it again.

                # Ensure audio data is float32, which is typically expected by deep learning models.
                if audio_input.dtype != np.float32:
                    audio_input = audio_input.astype(np.float32)
                
                # Normalize audio if its absolute maximum value exceeds 1.0, assuming it might be unnormalized
                # integer audio or similarly scaled float data. This brings it to the standard [-1.0, 1.0] range.
                if np.max(np.abs(audio_input)) > 1.0:
                    audio_input = audio_input / np.max(np.abs(audio_input))
                
                sf.write(audio_path, audio_input, sample_rate)
            else:
                raise TypeError(f"Unsupported `audio_input` type: {type(audio_input)}. "
                                "Expected `str`, `Path`, or `np.ndarray`.")
            
            # Perform the core separation using the internal Demucs method
            separated_tracks, output_sr = self._separate_with_demucs(audio_path)

            # If specific streams were requested during initialization, filter the results
            if self.streams:
                filtered_tracks = {stream: audio for stream, audio in separated_tracks.items() if stream in self.streams}
                # Provide a warning if none of the requested streams were found in the output
                if not filtered_tracks and self.streams:
                    print(f"Warning: None of the explicitly requested streams {self.streams} were found "
                          f"in the separated output. Available streams: {list(separated_tracks.keys())}")
                return filtered_tracks, output_sr
            
            return separated_tracks, output_sr

        # Catch and re-raise specific, expected exceptions for clearer error handling upstream
        except (FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
            raise
        except Exception as e:
            # Catch any other unexpected errors during the overall separation process
            print(f"An unexpected error occurred during source separation: {e}")
            raise RuntimeError(f"Source separation failed: {e}")
        finally:
            # Ensure the temporary audio file is cleaned up if it was created
            if temp_file_obj:
                try:
                    Path(temp_file_obj.name).unlink() # Delete the actual file from disk
                    # print(f"Temporary file '{audio_path}' cleaned up.") # For debugging
                except OSError as e:
                    print(f"Warning: Could not delete temporary file '{Path(temp_file_obj.name)}': {e}")
```