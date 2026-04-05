import os
import torch

class GeneralConfig:
    """General configuration settings for the system."""
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Seed for reproducibility
    SEED: int = 42

class SubjectManagerConfig:
    """Configuration for the Multi-Subject Latent Manager (MSLM) and its components."""
    # Latent Vector Pool
    MAX_SUBJECTS: int = 16  # Maximum number of subjects the pool can manage simultaneously
    LATENT_DIM: int = 768   # Dimension of each subject's latent state token
    LATENT_POOL_INITIALIZATION_STRATEGY: str = "random_normal" # Options: "random_normal", "zero", "learned_embedding"
    LATENT_POOL_REALLOCATION_BATCH_SIZE: int = 4 # How many new latent slots to allocate if pool grows

    # Subject Registry
    IDENTITY_FEATURE_DIM: int = 512 # Dimension of optional identity features (e.g., CLIP embeddings)
    ACTION_EMBEDDING_DIM: int = 128 # Dimension for embedding action descriptions (if used)
    # Coordinate format: [x, y, w, h] normalized to [0, 1]
    COORDINATE_NORMALIZATION_RANGE: tuple[float, float] = (0.0, 1.0) 

class SpatialBiasingConfig:
    """Configuration for the Spatial Biasing Adapter (SBA)."""
    BIASING_STRATEGY: str = "coordinate_aware_attention" # Options: "coordinate_aware_attention", "cross_attention", "film_modulation"
    
    # Common attention parameters for various strategies
    ATTENTION_HEADS: int = 8
    ATTENTION_DIM: int = 64 # Dimension per attention head (total dim will be ATTENTION_HEADS * ATTENTION_DIM)
    
    # Specific to Coordinate-Aware Attention
    SPATIAL_EMBEDDING_DIM: int = 128 # Dimension for spatial grid embeddings or positional encodings
    
    # Specific to Cross-Attention / FiLM Modulation
    # Defines at which U-Net resolutions (e.g., [64, 32, 16]) the adapter will inject biases.
    # The actual integration points depend on the specific U-Net architecture.
    U_NET_FEATURE_MAP_RESOLUTIONS: list[int] = [64, 32, 16] 
    
    # Weight for applying the biasing effect. Can be learned or fixed.
    BIAS_STRENGTH: float = 1.0 
    
    # Whether to use sparse attention mechanisms for efficiency
    USE_SPARSE_ATTENTION: bool = True
    SPARSE_ATTENTION_WINDOW_SIZE: int = 16 # Window size for local sparse attention (e.g., around subject)

class Config:
    """Aggregated configuration for the entire project."""
    GENERAL = GeneralConfig
    SUBJECT_MANAGER = SubjectManagerConfig
    SPATIAL_BIASING = SpatialBiasingConfig

    @staticmethod
    def print_config():
        """Prints the current configuration settings."""
        print("--- General Configuration ---")
        for attr, value in vars(Config.GENERAL).items():
            if not attr.startswith('__'):
                print(f"  {attr}: {value}")
        
        print("\n--- Subject Manager Configuration ---")
        for attr, value in vars(Config.SUBJECT_MANAGER).items():
            if not attr.startswith('__'):
                print(f"  {attr}: {value}")
        
        print("\n--- Spatial Biasing Configuration ---")
        for attr, value in vars(Config.SPATIAL_BIASING).items():
            if not attr.startswith('__'):
                print(f"  {attr}: {value}")

# Example usage (for verification/debugging)
if __name__ == "__main__":
    Config.print_config()
    print(f"\nRunning on device: {Config.GENERAL.DEVICE}")
    print(f"Max subjects: {Config.SUBJECT_MANAGER.MAX_SUBJECTS}")
    print(f"Biasing strategy: {Config.SPATIAL_BIASING.BIASING_STRATEGY}")