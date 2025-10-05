"""
Provider implementations for document extraction dependency injection.
Production and mock providers for configurable document text extraction.
"""

import logging
from pathlib import Path
from typing import Any

from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_error_context,
    log_performance_metric,
)

# ================================
# STANDARD PROVIDERS
# ================================


class ConfigProvider:
    """Configuration provider using TOML files."""

    def get_extraction_config(self) -> dict[str, Any]:
        """Get extraction configuration from TOML files."""
        logger = get_system_logger()
        log_component_start("config_provider", "get_extraction_config")

        try:
            from ..utils.config_loader import get_extraction_config

            logger.debug("config_provider", "get_extraction_config", "Loading extraction configuration")
            config = get_extraction_config()

            logger.debug(
                "config_provider", "get_extraction_config", f"Loaded extraction config with {len(config)} keys"
            )
            logger.trace("config_provider", "get_extraction_config", f"Extraction config keys: {list(config.keys())}")

            log_component_end("config_provider", "get_extraction_config", "Extraction config loaded")
            return config

        except Exception as e:
            log_error_context("config_provider", "get_extraction_config", e, {})
            raise


class FileSystemProvider:
    """File system provider using real file operations."""

    def file_exists(self, file_path: Path) -> bool:
        """Check if file exists."""
        logger = get_system_logger()
        logger.trace("filesystem_provider", "file_exists", f"Checking existence: {file_path}")

        exists = file_path.exists()
        logger.trace("filesystem_provider", "file_exists", f"File exists: {exists}")
        return exists

    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        logger = get_system_logger()
        log_component_start("filesystem_provider", "get_file_size_mb", file_path=str(file_path))

        try:
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)

            logger.debug("filesystem_provider", "get_file_size_mb", f"File size: {size_bytes} bytes = {size_mb:.2f} MB")
            log_performance_metric("filesystem_provider", "get_file_size_mb", "file_size_mb", size_mb)

            log_component_end("filesystem_provider", "get_file_size_mb", f"File size calculated: {size_mb:.2f} MB")
            return size_mb

        except Exception as e:
            log_error_context("filesystem_provider", "get_file_size_mb", e, {"file_path": str(file_path)})
            raise

    def open_binary(self, file_path: Path) -> bytes:
        """Open file in binary mode."""
        logger = get_system_logger()
        log_component_start("filesystem_provider", "open_binary", file_path=str(file_path))

        try:
            logger.debug("filesystem_provider", "open_binary", f"Reading binary file: {file_path}")
            content = file_path.read_bytes()

            logger.debug("filesystem_provider", "open_binary", f"Read {len(content)} bytes")
            log_performance_metric("filesystem_provider", "open_binary", "content_size_bytes", len(content))

            log_component_end("filesystem_provider", "open_binary", f"Binary file read: {len(content)} bytes")
            return content

        except Exception as e:
            log_error_context("filesystem_provider", "open_binary", e, {"file_path": str(file_path)})
            raise

    def open_text(self, file_path: Path, encoding: str) -> str:
        """Open file in text mode with specified encoding."""
        logger = get_system_logger()
        log_component_start("filesystem_provider", "open_text", file_path=str(file_path), encoding=encoding)

        try:
            logger.debug("filesystem_provider", "open_text", f"Reading text file with {encoding} encoding: {file_path}")
            content = file_path.read_text(encoding=encoding)

            logger.debug("filesystem_provider", "open_text", f"Read {len(content)} characters")
            log_performance_metric("filesystem_provider", "open_text", "content_size_chars", len(content))

            log_component_end("filesystem_provider", "open_text", f"Text file read: {len(content)} chars")
            return content

        except Exception as e:
            log_error_context(
                "filesystem_provider", "open_text", e, {"file_path": str(file_path), "encoding": encoding}
            )
            raise


class LoggerProvider:
    """Logger provider using Python logging."""

    def __init__(self, logger_name: str | None = None):
        """Initialize with logger name."""
        self.logger = logging.getLogger(logger_name or __name__)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)


# ================================
# FACTORY FUNCTIONS
# ================================


def create_config_provider(config_dict: dict | None = None):
    """Create configuration provider (supports mock dict for testing)."""
    if config_dict is not None:
        from tests.conftest import MockConfigProvider

        return MockConfigProvider(config_dict)
    return ConfigProvider()


def create_file_system_provider(mock: bool = False):
    """Create file system provider (supports mock mode for testing)."""
    if mock:
        from tests.conftest import MockFileSystemProvider

        return MockFileSystemProvider()
    return FileSystemProvider()


def create_logger_provider(logger_name: str | None = None, mock: bool = False):
    """Create logger provider (supports mock mode for testing)."""
    if mock:
        from tests.conftest import MockLoggerProvider

        return MockLoggerProvider()
    return LoggerProvider(logger_name)


def create_providers(logger_name: str | None = None):
    """
    Create full set of providers.

    Args:
        logger_name: Logger name for logging

    Returns:
        Tuple of (config_provider, file_system_provider, logger_provider)
    """
    config_provider = ConfigProvider()
    file_system_provider = FileSystemProvider()
    logger_provider = LoggerProvider(logger_name)

    return config_provider, file_system_provider, logger_provider
