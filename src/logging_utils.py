"""Logging helpers for the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Path, name: str, level: str = "INFO", console: bool = True) -> Path:
    """Configure logging to a file (and optionally console)."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger()
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler: Optional[logging.Handler] = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return log_file
