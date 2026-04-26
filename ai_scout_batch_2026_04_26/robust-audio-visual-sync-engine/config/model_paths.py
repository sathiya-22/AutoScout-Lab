import os

class ModelPaths:
    """
    Manages paths to pre-trained models used by the Robust Audio-Visual Synchronization Engine.
    Paths can be configured via environment variables or default to relative paths within
    the project's 'models/' directory.
    """

    # Base directory for all models. Can be overridden by the 'MODEL_BASE_DIR' environment variable.
    # Defaults to 'models/' relative to the project root.
    _DEFAULT_MODEL_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    MODEL_BASE_DIR = os.getenv('MODEL_BASE_DIR', _DEFAULT_MODEL_BASE_DIR)

    # Ensure the base directory exists
    if not os.path.exists(MODEL_BASE_DIR):
        try:
            os.makedirs(MODEL_BASE_DIR)
        except OSError as e:
            print(f"Warning: Could not create model base directory '{MODEL_BASE_DIR}': {e}")

    # Paths for Source Separation Models
    # Spleeter models often require specific subdirectories if downloaded via spleeter itself
    SPLEETER_MODEL_DIR = os.getenv(
        'SPLEETER_MODEL_DIR',
        os.path.join(MODEL_BASE_DIR, 'spleeter')
    )
    # Demucs models
    DEMUCS_MODEL_PATH = os.getenv(
        'DEMUCS_MODEL_PATH',
        os.path.join(MODEL_BASE_DIR, 'demucs', 'htdemucs.th') # Example for a specific Demucs model
    )

    # Paths for Pitch Tracking Models
    CREPE_MODEL_PATH = os.getenv(
        'CREPE_MODEL_PATH',
        os.path.join(MODEL_BASE_DIR, 'crepe', 'crepe_full.pth') # Example path for a CREPE model
    )
    WORLD_MODEL_PATH = os.getenv(
        'WORLD_MODEL_PATH',
        os.path.join(MODEL_BASE_DIR, 'world') # WORLD might involve multiple files or a library, this is a placeholder
    )

    # Paths for Audio-Visual Alignment Models (e.g., Cross-Modal Attention)
    CROSS_MODAL_ATTENTION_MODEL_PATH = os.getenv(
        'CROSS_MODAL_ATTENTION_MODEL_PATH',
        os.path.join(MODEL_BASE_DIR, 'av_alignment', 'cross_modal_attention_model.pth')
    )

    # Dictionary for easier access and iteration
    ALL_MODEL_PATHS = {
        "spleeter_dir": SPLEETER_MODEL_DIR,
        "demucs_model": DEMUCS_MODEL_PATH,
        "crepe_model": CREPE_MODEL_PATH,
        "world_dir": WORLD_MODEL_PATH,
        "cross_modal_attention_model": CROSS_MODAL_ATTENTION_MODEL_PATH,
    }

    @staticmethod
    def get_path(model_name: str) -> str:
        """
        Retrieves the path for a specific model by its logical name.

        Args:
            model_name (str): The logical name of the model (e.g., "crepe_model", "demucs_model").

        Returns:
            str: The file system path to the model.

        Raises:
            ValueError: If the model_name is not recognized.
        """
        path = ModelPaths.ALL_MODEL_PATHS.get(model_name)
        if path is None:
            raise ValueError(f"Unknown model name: '{model_name}'. Available models: {list(ModelPaths.ALL_MODEL_PATHS.keys())}")
        return path

    @staticmethod
    def print_paths():
        """Prints all configured model paths for debugging purposes."""
        print(f"--- Model Configuration ({os.path.basename(__file__)}) ---")
        print(f"  Base Model Directory: {ModelPaths.MODEL_BASE_DIR}")
        for name, path in ModelPaths.ALL_MODEL_PATHS.items():
            print(f"  {name.replace('_', ' ').title()}: {path}")
        print("---------------------------------------")


# Example of how to use this configuration:
if __name__ == '__main__':
    # You can set environment variables before running this script:
    # export MODEL_BASE_DIR="/mnt/data/my_models"
    # export CREPE_MODEL_PATH="/mnt/data/my_models/custom_crepe_tiny.pth"

    ModelPaths.print_paths()

    try:
        crepe_path = ModelPaths.get_path("crepe_model")
        print(f"\nAccessing CREPE model path: {crepe_path}")

        demucs_path = ModelPaths.get_path("demucs_model")
        print(f"Accessing Demucs model path: {demucs_path}")

        unknown_path = ModelPaths.get_path("non_existent_model")
        print(f"Accessing unknown model path: {unknown_path}")
    except ValueError as e:
        print(f"\nError accessing model: {e}")

    # Simulate model directories if they don't exist for the example
    # This is just for demonstration, in a real scenario, you'd download models there
    for name, path in ModelPaths.ALL_MODEL_PATHS.items():
        if name.endswith("_dir"): # Handle directories
            if not os.path.exists(path):
                print(f"Creating dummy directory: {path}")
                os.makedirs(path, exist_ok=True)
        else: # Handle files
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                print(f"Creating dummy parent directory for {path}: {parent_dir}")
                os.makedirs(parent_dir, exist_ok=True)
            if not os.path.exists(path):
                print(f"Creating dummy file: {path}")
                try:
                    with open(path, 'w') as f:
                        f.write(f"Dummy content for {name} model weights.")
                except IOError as e:
                    print(f"Error creating dummy file {path}: {e}")