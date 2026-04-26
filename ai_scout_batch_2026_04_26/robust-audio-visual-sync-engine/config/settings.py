import os

class Settings:
    # --- General Application Settings ---
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp/ravse_temp")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./ravse_output")

    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

    # --- Input Handling Settings ---
    ALLOWED_AUDIO_FORMATS: list[str] = os.getenv("ALLOWED_AUDIO_FORMATS", "mp3,wav,flac,m4a").split(',')
    ALLOWED_VIDEO_FORMATS: list[str] = os.getenv("ALLOWED_VIDEO_FORMATS", "mp4,avi,mov,mkv").split(',')
    VIDEO_DOWNGRADE_RESOLUTION: str = os.getenv("VIDEO_DOWNGRADE_RESOLUTION", "720p")

    # --- Audio Processing Settings ---
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
    AUDIO_CHUNK_SIZE_SEC: float = float(os.getenv("AUDIO_CHUNK_SIZE_SEC", "1.0"))
    AUDIO_MAX_DURATION_SEC: float = float(os.getenv("AUDIO_MAX_DURATION_SEC", "300.0"))
    
    # Source Separation
    SOURCE_SEPARATION_MODEL: str = os.getenv("SOURCE_SEPARATION_MODEL", "demucs")
    SOURCE_SEPARATION_SEGMENTS: list[str] = os.getenv("SOURCE_SEPARATION_SEGMENTS", "vocals,accompaniment").split(',')
    SOURCE_SEPARATION_GPU_ENABLED: bool = os.getenv("SOURCE_SEPARATION_GPU_ENABLED", "True").lower() == "true"
    
    # Pitch Tracking
    PITCH_TRACKING_ALGORITHM: str = os.getenv("PITCH_TRACKING_ALGORITHM", "crepe")
    PITCH_FRAME_LENGTH_MS: int = int(os.getenv("PITCH_FRAME_LENGTH_MS", "25"))
    PITCH_HOP_LENGTH_MS: int = int(os.getenv("PITCH_HOP_LENGTH_MS", "10"))
    PITCH_FMIN_HZ: float = float(os.getenv("PITCH_FMIN_HZ", "50.0"))
    PITCH_FMAX_HZ: float = float(os.getenv("PITCH_FMAX_HZ", "1500.0"))
    PITCH_MODEL_CAPACITY: str = os.getenv("PITCH_MODEL_CAPACITY", "full")
    NOISE_REDUCTION_ENABLED: bool = os.getenv("NOISE_REDUCTION_ENABLED", "True").lower() == "true"

    # Feature Extraction
    MFCC_N_MFCC: int = int(os.getenv("MFCC_N_MFCC", "20"))
    SPECTROGRAM_N_FFT: int = int(os.getenv("SPECTROGRAM_N_FFT", "2048"))
    SPECTROGRAM_HOP_LENGTH: int = int(os.getenv("SPECTROGRAM_HOP_LENGTH", "512"))

    # --- Audio-Visual Alignment Settings ---
    ALIGNMENT_STRATEGY: str = os.getenv("ALIGNMENT_STRATEGY", "hybrid")
    
    # Cross-Modal Attention
    CROSS_MODAL_ATTENTION_MODEL_PATH: str = os.getenv("CROSS_MODAL_ATTENTION_MODEL_PATH", "models/av_attention/best_model.pth")
    ATTENTION_EMBEDDING_DIM: int = int(os.getenv("ATTENTION_EMBEDDING_DIM", "512"))
    ATTENTION_NUM_HEADS: int = int(os.getenv("ATTENTION_NUM_HEADS", "8"))
    ATTENTION_LEARNING_RATE: float = float(os.getenv("ATTENTION_LEARNING_RATE", "1e-4"))
    
    # Correlation Analysis
    CORRELATION_WINDOW_SEC: float = float(os.getenv("CORRELATION_WINDOW_SEC", "5.0"))
    CORRELATION_OVERLAP_SEC: float = float(os.getenv("CORRELATION_OVERLAP_SEC", "1.0"))
    DTW_RADIUS: int = int(os.getenv("DTW_RADIUS", "3"))

    # --- Confidence Scoring Settings ---
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    
    SIGNAL_CLARITY_WEIGHT: float = float(os.getenv("SIGNAL_CLARITY_WEIGHT", "0.3"))
    PITCH_ROBUSTNESS_WEIGHT: float = float(os.getenv("PITCH_ROBUSTNESS_WEIGHT", "0.4"))
    ALIGNMENT_AGREEMENT_WEIGHT: float = float(os.getenv("ALIGNMENT_AGREEMENT_WEIGHT", "0.3"))
    
    # --- Model Paths (relative to project root or absolute) ---
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
    
    DEMUCS_MODEL_ID: str = os.getenv("DEMUCS_MODEL_ID", "htdemucs")
    SPLEETER_MODEL_ID: str = os.getenv("SPLEETER_MODEL_ID", "2stems")
    CREPE_MODEL_PATH: str = os.path.join(MODELS_DIR, "crepe", f"crepe_{PITCH_MODEL_CAPACITY}.pth")
    WORLD_BIN_PATH: str = os.path.join(MODELS_DIR, "world", "bin")

    # --- API Settings (for web_service.py) ---
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "5"))
    MAX_STREAMING_CHUNK_DURATION_SEC: float = float(os.getenv("MAX_STREAMING_CHUNK_DURATION_SEC", "10.0"))

Settings.ensure_dirs()