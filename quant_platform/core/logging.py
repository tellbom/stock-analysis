"""
core.logging
============
Thin wrapper around stdlib logging.  All modules in quant_platform use
``get_logger(__name__)`` so output is consistent and redirectable.

Kept intentionally minimal — no third-party deps, no over-engineering.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent format; safe to call multiple times."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    return logger
