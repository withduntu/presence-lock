"""
Writes timestamped event lines to a log file (and echoes them to the
console). Only ever logs short text descriptions of events - never frames
or images, per the project's local-processing-only rule.
"""

import logging
import os


def setup_logger(log_file: str) -> logging.Logger:
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("presence_lock")
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # avoid duplicate handlers if called more than once
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
