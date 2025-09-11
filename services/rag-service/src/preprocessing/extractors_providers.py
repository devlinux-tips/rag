"""
Provider implementations for document extraction dependency injection.
Production and mock providers for configurable document text extraction.
"""

import logging
from pathlib import Path
from typing import Any, Dict

# ================================
# PRODUCTION PROVIDERS
# ================================


class ProductionConfigProvider:
    """Production configuration provider using real TOML files."""

    def get_extraction_config(self) -> Dict[str, Any]:
        """Get extraction configuration from TOML files."""
        from ..utils.config_loader import get_extraction_config

        return get_extraction_config()


class ProductionFileSystemProvider:
    """Production file system provider using real file operations."""

    def file_exists(self, file_path: Path) -> bool:
        """Check if file exists."""
        return file_path.exists()

    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        return file_path.stat().st_size / (1024 * 1024)

    def open_binary(self, file_path: Path) -> bytes:
        """Open file in binary mode."""
        return file_path.read_bytes()

    def open_text(self, file_path: Path, encoding: str) -> str:
        """Open file in text mode with specified encoding."""
        return file_path.read_text(encoding=encoding)


class ProductionLoggerProvider:
    """Production logger provider using actual Python logging."""

    def __init__(self, logger_name: str = None):
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

    def __init__(self, extraction_config: Dict[str, Any]):
        """Initialize with extraction configuration."""
        self.extraction_config = extraction_config

    def get_extraction_config(self) -> Dict[str, Any]:
        """Get mock extraction configuration."""
        return self.extraction_config


class MockFileSystemProvider:
    """Mock file system provider for testing."""

    def __init__(self):
        """Initialize mock file system."""
        self.files: Dict[str, bytes] = {}
        self.file_sizes_mb: Dict[str, float] = {}
        self.existing_files: set = set()

    def add_file(self, file_path: str, content: bytes, size_mb: float = None) -> None:
        """Add a mock file to the file system."""
        self.files[file_path] = content
        self.file_sizes_mb[file_path] = size_mb or len(content) / (1024 * 1024)
        self.existing_files.add(file_path)

    def add_text_file(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        size_mb: float = None,
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

    def get_all_messages(self) -> Dict[str, list]:
        """Get all logged messages for testing."""
        return {"info": self.info_messages, "error": self.error_messages}

    def clear_messages(self) -> None:
        """Clear all logged messages."""
        self.info_messages.clear()
        self.error_messages.clear()


# ================================
# FACTORY FUNCTIONS
# ================================


def create_config_provider(mock_config: Dict[str, Any] = None):
    """Create configuration provider (production or mock)."""
    if mock_config is not None:
        return MockConfigProvider(mock_config)
    return ProductionConfigProvider()


def create_file_system_provider(mock_files: Dict[str, bytes] = None):
    """Create file system provider (production or mock)."""
    if mock_files is not None:
        provider = MockFileSystemProvider()
        for file_path, content in mock_files.items():
            provider.add_file(file_path, content)
        return provider
    return ProductionFileSystemProvider()


def create_logger_provider(logger_name: str = None, mock: bool = False):
    """Create logger provider (production or mock)."""
    if mock:
        return MockLoggerProvider()
    return ProductionLoggerProvider(logger_name)


# ================================
# CONVENIENCE PROVIDER BUILDERS
# ================================


def create_test_providers(
    config: Dict[str, Any] = None,
    files: Dict[str, bytes] = None,
    mock_logging: bool = True,
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


def create_production_providers(logger_name: str = None):
    """
    Create full set of production providers.

    Args:
        logger_name: Logger name for production logging

    Returns:
        Tuple of (config_provider, file_system_provider, logger_provider)
    """
    config_provider = ProductionConfigProvider()
    file_system_provider = ProductionFileSystemProvider()
    logger_provider = ProductionLoggerProvider(logger_name)

    return config_provider, file_system_provider, logger_provider
