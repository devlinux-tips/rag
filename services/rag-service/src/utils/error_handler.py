"""
Error handling utilities for configuration loading.
Provides consistent logging and fallback behavior across modules.
"""

import logging
from typing import Any, Callable, Optional


def handle_config_error(
    operation: Callable[[], Any],
    fallback_value: Any,
    config_file: str,
    section: str,
    logger: Optional[logging.Logger] = None,
    error_level: str = "warning",
) -> Any:
    """
    Handle configuration loading with consistent error logging and fallback.

    Args:
        operation: Function that loads the config (e.g., lambda: get_embeddings_config())
        fallback_value: Value to return if config loading fails
        config_file: Name of the config file (e.g., "config/vectordb.toml")
        section: Config section name (e.g., "[embeddings]", "[search.weights]")
        logger: Logger instance (creates one if None)
        error_level: Log level for config failures ("error", "warning", "info")

    Returns:
        Config value from operation() or fallback_value if loading fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        return operation()
    except Exception as e:
        # Log the specific error
        if error_level == "error":
            logger.error(f"Failed to load config from {config_file}: {e}")
        elif error_level == "warning":
            logger.warning(f"Failed to load config from {config_file}: {e}")
        else:
            logger.info(f"Failed to load config from {config_file}: {e}")

        # Log fallback guidance
        logger.warning(f"Using hardcoded fallback. Check {section} section.")

        return fallback_value


def create_config_loader(config_file: str, logger_name: str = None):
    """
    Create a specialized config loader for a specific file.

    Args:
        config_file: Config file name (e.g., "config/vectordb.toml")
        logger_name: Logger name (uses config_file if None)

    Returns:
        Function that handles config loading with pre-set file name
    """
    if logger_name is None:
        logger_name = config_file.replace("/", ".").replace(".toml", "")

    logger = logging.getLogger(logger_name)

    def load_with_fallback(
        operation: Callable[[], Any],
        fallback_value: Any,
        section: str,
        error_level: str = "warning",
    ) -> Any:
        """Pre-configured loader for this config file."""
        return handle_config_error(
            operation=operation,
            fallback_value=fallback_value,
            config_file=config_file,
            section=section,
            logger=logger,
            error_level=error_level,
        )

    return load_with_fallback
