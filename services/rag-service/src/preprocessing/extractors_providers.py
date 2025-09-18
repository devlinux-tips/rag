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
# MOCK PROVIDERS FOR TESTING
# ================================


class MockConfigProvider:
    """Mock configuration provider for testing."""

    def __init__(self, extraction_config: dict[str, Any]):
        """Initialize with extraction configuration."""
        self.extraction_config = extraction_config

    def get_extraction_config(self) -> dict[str, Any]:
        """Get mock extraction configuration."""
        return self.extraction_config


class MockFileSystemProvider:
    """Mock file system provider for testing."""

    def __init__(self):
        """Initialize mock file system."""
        self.files: dict[str, bytes] = {}
        self.file_sizes_mb: dict[str, float] = {}
        self.existing_files: set = set()

    def add_file(self, file_path: str, content: bytes, size_mb: float | None = None) -> None:
        """Add a mock file to the file system."""
        self.files[file_path] = content
        self.file_sizes_mb[file_path] = size_mb or len(content) / (1024 * 1024)
        self.existing_files.add(file_path)

    def add_text_file(
        self, file_path: str, content: str, encoding: str = "utf-8", size_mb: float | None = None
    ) -> None:
        """Add a mock text file to the file system."""
        binary_content = content.encode(encoding)
        self.add_file(file_path, binary_content, size_mb)

    def file_exists(self, file_path: Path) -> bool:
        """Check if mock file exists."""
        return str(file_path) in self.existing_files

    def get_file_size_mb(self, file_path: Path) -> float:
        """Get mock file size in MB."""
        path_str = str(file_path)
        if path_str not in self.file_sizes_mb:
            raise FileNotFoundError(f"Mock file not found: {file_path}")
        return self.file_sizes_mb[path_str]

    def open_binary(self, file_path: Path) -> bytes:
        """Open mock file in binary mode."""
        path_str = str(file_path)
        if path_str not in self.files:
            raise FileNotFoundError(f"Mock file not found: {file_path}")
        return self.files[path_str]

    def open_text(self, file_path: Path, encoding: str) -> str:
        """Open mock file in text mode with specified encoding."""
        binary_content = self.open_binary(file_path)
        return binary_content.decode(encoding)


class MockLoggerProvider:
    """Mock logger provider for testing."""

    def __init__(self):
        """Initialize mock logger."""
        self.info_messages = []
        self.error_messages = []

    def info(self, message: str) -> None:
        """Record info message."""
        self.info_messages.append(message)

    def error(self, message: str) -> None:
        """Record error message."""
        self.error_messages.append(message)

    def get_all_messages(self) -> dict[str, list]:
        """Get all logged messages for testing."""
        return {"info": self.info_messages, "error": self.error_messages}

    def clear_messages(self) -> None:
        """Clear all logged messages."""
        self.info_messages.clear()
        self.error_messages.clear()


# ================================
# FACTORY FUNCTIONS
# ================================


def create_config_provider(mock_config: dict[str, Any] | None = None):
    """Create configuration provider (real or mock)."""
    if mock_config is not None:
        return MockConfigProvider(mock_config)
    return ConfigProvider()


def create_file_system_provider(mock_files: dict[str, bytes] | None = None):
    """Create file system provider (real or mock)."""
    if mock_files is not None:
        provider = MockFileSystemProvider()
        for file_path, content in mock_files.items():
            provider.add_file(file_path, content)
        return provider
    return FileSystemProvider()


def create_logger_provider(logger_name: str | None = None, mock: bool = False):
    """Create logger provider (real or mock)."""
    if mock:
        return MockLoggerProvider()
    return LoggerProvider(logger_name)


# ================================
# CONVENIENCE PROVIDER BUILDERS
# ================================


def create_test_providers(
    config: dict[str, Any] | None = None, files: dict[str, bytes] | None = None, mock_logging: bool = True
):
    """
    Create full set of test providers for testing.

    Args:
        config: Mock extraction configuration
        files: Mock files as {path: content} mapping
        mock_logging: Whether to use mock logging

    Returns:
        Tuple of (config_provider, file_system_provider, logger_provider)
    """
    # Default test configuration
    default_config = {
        "supported_formats": [".pdf", ".docx", ".txt"],
        "text_encodings": ["utf-8", "latin1", "cp1252"],
        "max_file_size_mb": 50,
        "enable_logging": True,
    }

    config_provider = create_config_provider(config or default_config)
    file_system_provider = create_file_system_provider(files)
    logger_provider = create_logger_provider(mock=mock_logging)

    return config_provider, file_system_provider, logger_provider


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
