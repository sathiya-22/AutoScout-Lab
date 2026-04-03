import logging
import os

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.debug(f"Initializing src package, version {__version__}")