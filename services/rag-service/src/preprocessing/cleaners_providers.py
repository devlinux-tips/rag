"""
Provider implementations for multilingual text cleaning dependency injection.
Standard and mock providers for configurable text cleaning operations.
"""

import locale
import logging
import os
from typing import Any

from ..utils.logging_factory import get_system_logger, log_component_end, log_component_start, log_error_context

# ================================
# STANDARD PROVIDERS
# ================================


class ConfigProvider:
    """Standard provider using real TOML files."""

    def get_language_config(self, language: str) -> dict[str, Any]:
        """Get language-specific text processing configuration."""
        logger = get_system_logger()
        log_component_start("config_provider", "get_language_config", language=language)

        try:
            from ..utils.config_loader import get_language_specific_config

            logger.debug("config_provider", "get_language_config", f"Loading text_processing config for {language}")
            config = get_language_specific_config("text_processing", language)

            logger.debug("config_provider", "get_language_config", f"Loaded config with {len(config)} keys")
            logger.trace("config_provider", "get_language_config", f"Config keys: {list(config.keys())}")

            log_component_end("config_provider", "get_language_config", f"Language config loaded for {language}")
            return config

        except Exception as e:
            log_error_context("config_provider", "get_language_config", e, {"language": language})
            raise

    def get_cleaning_config(self) -> dict[str, Any]:
        """Get general cleaning configuration."""
        logger = get_system_logger()
        log_component_start("config_provider", "get_cleaning_config")

        try:
            from ..utils.config_loader import get_cleaning_config

            logger.debug("config_provider", "get_cleaning_config", "Loading general cleaning configuration")
            config = get_cleaning_config()

            logger.debug("config_provider", "get_cleaning_config", f"Loaded cleaning config with {len(config)} keys")
            logger.trace("config_provider", "get_cleaning_config", f"Cleaning config keys: {list(config.keys())}")

            log_component_end("config_provider", "get_cleaning_config", "General cleaning config loaded")
            return config

        except Exception as e:
            log_error_context("config_provider", "get_cleaning_config", e, {})
            raise

    def get_document_cleaning_config(self, language: str) -> dict[str, Any]:
        """Get document cleaning configuration."""
        logger = get_system_logger()
        log_component_start("config_provider", "get_document_cleaning_config", language=language)

        try:
            from ..utils.config_loader import get_language_specific_config

            logger.debug(
                "config_provider", "get_document_cleaning_config", f"Loading document_cleaning config for {language}"
            )
            config = get_language_specific_config("document_cleaning", language)

            logger.debug(
                "config_provider",
                "get_document_cleaning_config",
                f"Loaded document cleaning config with {len(config)} keys",
            )
            logger.trace(
                "config_provider",
                "get_document_cleaning_config",
                f"Document cleaning config keys: {list(config.keys())}",
            )

            log_component_end(
                "config_provider", "get_document_cleaning_config", f"Document cleaning config loaded for {language}"
            )
            return config

        except Exception as e:
            log_error_context("config_provider", "get_document_cleaning_config", e, {"language": language})
            raise

    def get_chunking_config(self, language: str) -> dict[str, Any]:
        """Get chunking configuration (merged)."""
        logger = get_system_logger()
        log_component_start("config_provider", "get_chunking_config", language=language)

        try:
            from ..utils.config_loader import get_chunking_config, get_language_specific_config

            logger.debug("config_provider", "get_chunking_config", "Loading main chunking configuration")
            main_chunking_config = get_chunking_config()

            logger.debug(
                "config_provider", "get_chunking_config", f"Loading language-specific chunking config for {language}"
            )
            language_chunking_config = get_language_specific_config("chunking", language)

            logger.debug(
                "config_provider",
                "get_chunking_config",
                f"Merging configs: {len(main_chunking_config)} main + {len(language_chunking_config)} language",
            )
            merged_config = {**main_chunking_config, **language_chunking_config}

            logger.debug(
                "config_provider", "get_chunking_config", f"Merged chunking config has {len(merged_config)} keys"
            )
            logger.trace("config_provider", "get_chunking_config", f"Merged config keys: {list(merged_config.keys())}")

            log_component_end("config_provider", "get_chunking_config", f"Chunking config merged for {language}")
            return merged_config

        except Exception as e:
            log_error_context("config_provider", "get_chunking_config", e, {"language": language})
            raise

    def get_shared_language_config(self, language: str) -> dict[str, Any]:
        """Get shared language configuration."""
        logger = get_system_logger()
        log_component_start("config_provider", "get_shared_language_config", language=language)

        try:
            from ..utils.config_loader import get_language_shared

            logger.debug(
                "config_provider", "get_shared_language_config", f"Loading shared language config for {language}"
            )
            config = get_language_shared(language)

            logger.debug(
                "config_provider", "get_shared_language_config", f"Loaded shared config with {len(config)} keys"
            )
            logger.trace("config_provider", "get_shared_language_config", f"Shared config keys: {list(config.keys())}")

            log_component_end(
                "config_provider", "get_shared_language_config", f"Shared language config loaded for {language}"
            )
            return config

        except Exception as e:
            log_error_context("config_provider", "get_shared_language_config", e, {"language": language})
            raise


class LoggerProvider:
    """Standard provider using actual Python logging."""

    def __init__(self, logger_name: str | None = None):
        """Initialize with logger name."""
        self.logger = logging.getLogger(logger_name or __name__)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)


class EnvironmentProvider:
    """Standard provider using real system operations."""

    def set_environment_variable(self, key: str, value: str) -> None:
        """Set environment variable."""
        os.environ[key] = value

    def set_locale(self, category: int, locale_name: str) -> None:
        """Set locale."""
        locale.setlocale(category, locale_name)


# ================================
# FACTORY FUNCTIONS
# ================================


def create_config_provider(config_dict: dict | None = None):
    """Create configuration provider (supports mock dict for testing)."""
    if config_dict is not None:
        from tests.conftest import MockConfigProvider

        return MockConfigProvider(config_dict)
    return ConfigProvider()


def create_logger_provider(logger_name: str | None = None, mock: bool = False):
    """Create logger provider (supports mock mode for testing)."""
    if mock:
        from tests.conftest import MockLoggerProvider

        return MockLoggerProvider()
    return LoggerProvider(logger_name)


def create_environment_provider(mock: bool = False):
    """Create environment provider (supports mock mode for testing)."""
    if mock:
        from tests.conftest import MockEnvironmentProvider

        return MockEnvironmentProvider()
    return EnvironmentProvider()


def create_providers(logger_name: str | None = None):
    """
    Create full set of providers.

    Args:
        logger_name: Logger name for logging

    Returns:
        Tuple of (config_provider, logger_provider, environment_provider)
    """
    config_provider = ConfigProvider()
    logger_provider = LoggerProvider(logger_name)
    environment_provider = EnvironmentProvider()

    return config_provider, logger_provider, environment_provider
