import logging
import os
import datetime

_logging_setup_complete = False

def setup_logging(
    log_file_path: str = None,
    log_level: str = None,
    console_output: bool = True,
    file_output: bool = True,
    propagate_to_root: bool = False
):
    global _logging_setup_complete
    if _logging_setup_complete:
        return

    if log_level is None:
        log_level = os.environ.get("PROTOTYPE_LOG_LEVEL", "INFO").upper()

    numeric_log_level = getattr(logging, log_level, logging.INFO)
    if not isinstance(numeric_log_level, int):
        print(f"WARNING: Invalid log level '{log_level}' specified. Falling back to INFO.", file=os.sys.stderr)
        numeric_log_level = logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_log_level)
    root_logger.propagate = propagate_to_root

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_log_level)
        root_logger.addHandler(console_handler)

    if file_output:
        if log_file_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(script_dir, '..', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(log_dir, f"prototype_{timestamp}.log")

        try:
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_log_level)
            root_logger.addHandler(file_handler)
            if console_output: # Only log this to console if console output is enabled
                root_logger.info(f"Logging output also directed to file: {log_file_path}")
            else: # If no console output, print to stderr directly
                print(f"INFO: Logging output directed to file: {log_file_path}", file=os.sys.stderr)
        except IOError as e:
            root_logger.error(f"Failed to set up file logging at {log_file_path}: {e}", exc_info=False)
            if not console_output:
                print(f"ERROR: Logging to file failed at {log_file_path}: {e}. No console output enabled.", file=os.sys.stderr)

    _logging_setup_complete = True