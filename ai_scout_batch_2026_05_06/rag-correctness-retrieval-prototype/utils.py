import os
import logging
import time
import functools
import tiktoken

# Setup a logger for this utility file itself, or the main application can configure it.
# For now, a basic setup.
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configures the root logger for the application.
    Args:
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (str, optional): Path to a log file. If None, logs only to console.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to prevent duplicate logs if called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except OSError as e:
            root_logger.error(f"Could not set up file logging to {log_file}: {e}")

    root_logger.info(f"Logging set up with level {logging.getLevelName(log_level)}" + (f" to file {log_file}" if log_file else ""))


def create_directory_if_not_exists(path: str):
    """
    Creates a directory if it does not already exist.

    Args:
        path (str): The path to the directory to create.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory: {path}")
        else:
            logger.debug(f"Directory already exists: {path}")
    except OSError as e:
        logger.error(f"Error creating directory {path}: {e}")
        raise # Re-raise to ensure calling code knows about the failure

def retry(max_retries: int = 3, delay_seconds: float = 1.0, backoff_factor: float = 2.0,
          exceptions=(Exception,)):
    """
    A decorator for retrying a function multiple times with exponential backoff.

    Args:
        max_retries (int): Maximum number of retries.
        delay_seconds (float): Initial delay in seconds before the first retry.
        backoff_factor (float): Factor by which the delay increases after each retry.
        exceptions (tuple): A tuple of exception types to catch and retry on.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay_seconds
            for i in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if i == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    logger.warning(
                        f"Attempt {i+1}/{max_retries+1} for {func.__name__} failed: {e}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
        return wrapper
    return decorator

def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Counts the number of tokens in a given text using tiktoken.

    Args:
        text (str): The input string to count tokens for.
        model_name (str): The name of the LLM model to use for tokenization
                          (e.g., "gpt-4", "gpt-3.5-turbo").

    Returns:
        int: The number of tokens in the text.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except KeyError:
        logger.warning(f"Model '{model_name}' not found for tiktoken encoding. Falling back to `cl100k_base` encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"An error occurred while counting tokens: {e}")
        return 0 # Return 0 or raise, depending on desired error handling policy