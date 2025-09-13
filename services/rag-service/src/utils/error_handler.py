"""
Error handling utilities for the RAG system.
Provides fail-fast error handling following AI_INSTRUCTIONS.md governance.
"""

import logging
from typing import Optional


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance with consistent configuration.

    Args:
        name: Logger name (uses __name__ if None)

    Returns:
        Configured logger instance
    """
    logger_name = name or __name__
    return logging.getLogger(logger_name)
