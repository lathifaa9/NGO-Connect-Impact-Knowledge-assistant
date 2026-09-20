"""
Logging module for NGO Connect & Impact Knowledge Assistant.
Provides standardized, structured console and file logging.
"""

import logging
import sys
from pathlib import Path

def setup_logger(name: str = "ngo_assistant", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()
