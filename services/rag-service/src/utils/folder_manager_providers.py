"""
Provider implementations for folder manager dependency injection.
Production and mock providers for testable folder management system.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from .folder_manager import (
    ConfigProvider,
    FileSystemProvider,
    FolderConfig,
    FolderStats,
    LoggerProvider,
)

# ================================
# MOCK PROVIDERS FOR TESTING
# ================================


class MockFileSystemProvider:
    """Mock filesystem provider for testing."""

    def __init__(self):
        """Initialize with in-memory filesystem simulation."""
        self.created_folders: List[str] = []
        self.existing_folders: Dict[str, bool] = {}
        self.folder_stats: Dict[str, FolderStats] = {}
        self.call_history: List[Dict] = []
        self.should_fail: Dict[str, bool] = {}

    def set_folder_exists(self, folder_path: Path, exists: bool = True) -> None:
        """Set whether a folder should be considered to exist."""
        self.existing_folders[str(folder_path)] = exists

    def set_folder_stats(self, folder_path: Path, stats: FolderStats) -> None:
        """Set mock statistics for a folder."""
        self.folder_stats[str(folder_path)] = stats

    def set_should_fail(self, operation: str, should_fail: bool = True) -> None:
        """Set whether an operation should fail."""
        self.should_fail[operation] = should_fail

    def create_folder(self, folder_path: Path) -> bool:
        """Mock folder creation."""
        self.call_history.append({"operation": "create_folder", "path": str(folder_path)})

        if self.should_fail.get("create_folder", False):
            return False

        path_str = str(folder_path)
        if path_str not in [str(p) for p in self.created_folders]:
            self.created_folders.append(path_str)
            self.existing_folders[path_str] = True
            return True
        return False

    def folder_exists(self, folder_path: Path) -> bool:
        """Mock folder existence check."""
        self.call_history.append({"operation": "folder_exists", "path": str(folder_path)})
        path_str = str(folder_path)
        if path_str not in self.existing_folders:
            raise ValueError(f"Mock folder existence not configured for {folder_path}")
        return self.existing_folders[path_str]

    def remove_folder(self, folder_path: Path) -> bool:
        """Mock folder removal."""
        self.call_history.append({"operation": "remove_folder", "path": str(folder_path)})

        if self.should_fail.get("remove_folder", False):
            return False

        path_str = str(folder_path)
        if path_str in self.existing_folders:
            del self.existing_folders[path_str]
            if path_str in self.created_folders:
                self.created_folders.remove(path_str)
        return True

    def get_folder_stats(self, folder_path: Path) -> FolderStats:
        """Mock folder statistics."""
        self.call_history.append({"operation": "get_folder_stats", "path": str(folder_path)})

        path_str = str(folder_path)
        if path_str not in self.folder_stats:
            raise ValueError(f"Mock folder stats not configured for {folder_path}")
        return self.folder_stats[path_str]

    def clear_history(self) -> None:
        """Clear operation history."""
        self.call_history.clear()

    def get_created_folders(self) -> List[str]:
        """Get list of folders that were created."""
        return self.created_folders.copy()


class MockConfigProvider:
    """Mock configuration provider for testing."""

    def __init__(self, config: Optional[FolderConfig] = None):
        """Initialize with optional mock configuration."""
        self.config = config or self._create_default_config()
        self.call_history: List[str] = []

    def _create_default_config(self) -> FolderConfig:
        """Create default test configuration."""
        return FolderConfig(
            data_base_dir="/mock/data",
            models_base_dir="/mock/models",
            system_dir="/mock/system",
            tenant_root_template="{data_base_dir}/tenants/{tenant_slug}",
            user_documents_template="{data_base_dir}/tenants/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/tenants/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/tenants/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/tenants/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/vectordb/{tenant_slug}",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}",
        )

    def set_config(self, config: FolderConfig) -> None:
        """Set mock configuration."""
        self.config = config

    def get_folder_config(self) -> FolderConfig:
        """Get folder configuration."""
        self.call_history.append("get_folder_config")
        return self.config


class MockLoggerProvider:
    """Mock logger provider that captures messages for testing."""

    def __init__(self):
        """Initialize message capture."""
        self.messages: Dict[str, List[str]] = {
            "info": [],
            "debug": [],
            "warning": [],
            "error": [],
        }

    def info(self, message: str) -> None:
        """Capture info message."""
        self.messages["info"].append(message)

    def debug(self, message: str) -> None:
        """Capture debug message."""
        self.messages["debug"].append(message)

    def warning(self, message: str) -> None:
        """Capture warning message."""
        self.messages["warning"].append(message)

    def error(self, message: str) -> None:
        """Capture error message."""
        self.messages["error"].append(message)

    def clear_messages(self) -> None:
        """Clear all captured messages."""
        for level in self.messages:
            self.messages[level].clear()

    def get_messages(self, level: str = None) -> Dict[str, List[str]] | List[str]:
        """Get captured messages by level or all messages."""
        if level:
            if level not in self.messages:
                raise ValueError(f"Unknown log level: {level}")
            return self.messages[level]
        return self.messages


# ================================
# PRODUCTION PROVIDERS
# ================================


class ProductionFileSystemProvider:
    """Production filesystem provider using real filesystem operations."""

    def __init__(self):
        """Initialize production filesystem provider."""
        self.logger = logging.getLogger(__name__)

    def create_folder(self, folder_path: Path) -> bool:
        """Create a folder if it doesn't exist."""
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created folder: {folder_path}")
            return True
        return False

    def folder_exists(self, folder_path: Path) -> bool:
        """Check if folder exists."""
        return folder_path.exists() and folder_path.is_dir()

    def remove_folder(self, folder_path: Path) -> bool:
        """Remove folder and all contents."""
        if folder_path.exists():
            shutil.rmtree(folder_path)
            self.logger.info(f"Removed folder: {folder_path}")
            return True
        return False

    def get_folder_stats(self, folder_path: Path) -> FolderStats:
        """Get file count and size statistics for folder."""
        stats = FolderStats(count=0, size_bytes=0)

        if folder_path.exists():
            for file_path in folder_path.rglob("*"):
                if file_path.is_file():
                    stats.count += 1
                    stats.size_bytes += file_path.stat().st_size

        return stats


class ProductionConfigProvider:
    """Production configuration provider using real config system."""

    def __init__(self):
        """Initialize production config provider."""
        self._config_cache: Optional[FolderConfig] = None

    def get_folder_config(self) -> FolderConfig:
        """Get folder configuration from real config system."""
        if self._config_cache is None:
            self._config_cache = self._load_config_from_system()
        return self._config_cache

    def _load_config_from_system(self) -> FolderConfig:
        """Load configuration from the real system."""
        try:
            # Import at runtime to avoid circular dependencies
            from ..utils.config_loader import get_shared_config

            config = get_shared_config()

            return FolderConfig(
                data_base_dir=config["data_base_dir"],
                models_base_dir=config["models_base_dir"],
                system_dir=config["system_dir"],
                tenant_root_template=config["tenant_root_template"],
                user_documents_template=config["user_documents_template"],
                tenant_shared_template=config["tenant_shared_template"],
                user_processed_template=config["user_processed_template"],
                tenant_processed_template=config["tenant_processed_template"],
                chromadb_path_template=config["chromadb_path_template"],
                models_path_template=config["models_path_template"],
                collection_name_template=config["collection_name_template"],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load folder configuration from system: {e}")


class StandardLoggerProvider:
    """Standard logger provider using Python's logging system."""

    def __init__(self, logger_name: str = __name__):
        """Initialize with logger."""
        self.logger = logging.getLogger(logger_name)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)


# ================================
# CONVENIENCE FACTORY FUNCTIONS
# ================================


def create_mock_setup(
    config: Optional[FolderConfig] = None,
    existing_folders: Optional[Dict[str, bool]] = None,
    folder_stats: Optional[Dict[str, FolderStats]] = None,
    filesystem_failures: Optional[Dict[str, bool]] = None,
) -> tuple:
    """
    Create complete mock setup for testing.

    Args:
        config: Optional mock configuration
        existing_folders: Optional dict of folder existence states
        folder_stats: Optional dict of folder statistics
        filesystem_failures: Optional dict of operations that should fail

    Returns:
        Tuple of (config_provider, filesystem_provider, logger_provider)
    """
    # Create mock components
    config_provider = MockConfigProvider(config)
    filesystem_provider = MockFileSystemProvider()
    logger_provider = MockLoggerProvider()

    # Configure existing folders
    if existing_folders:
        for folder_path, exists in existing_folders.items():
            filesystem_provider.set_folder_exists(Path(folder_path), exists)

    # Configure folder stats
    if folder_stats:
        for folder_path, stats in folder_stats.items():
            filesystem_provider.set_folder_stats(Path(folder_path), stats)

    # Configure filesystem failures
    if filesystem_failures:
        for operation, should_fail in filesystem_failures.items():
            filesystem_provider.set_should_fail(operation, should_fail)

    return config_provider, filesystem_provider, logger_provider


def create_production_setup(logger_name: Optional[str] = None) -> tuple:
    """
    Create production setup with real components.

    Args:
        logger_name: Optional logger name override

    Returns:
        Tuple of (config_provider, filesystem_provider, logger_provider)
    """
    config_provider = ProductionConfigProvider()
    filesystem_provider = ProductionFileSystemProvider()
    logger_provider = StandardLoggerProvider(logger_name or __name__)

    return config_provider, filesystem_provider, logger_provider


def create_test_config(
    data_base_dir: str = "/test/data",
    models_base_dir: str = "/test/models",
    system_dir: str = "/test/system",
) -> FolderConfig:
    """Create test configuration with customizable parameters."""
    return FolderConfig(
        data_base_dir=data_base_dir,
        models_base_dir=models_base_dir,
        system_dir=system_dir,
        tenant_root_template="{data_base_dir}/tenants/{tenant_slug}",
        user_documents_template="{data_base_dir}/tenants/{tenant_slug}/users/{user_id}/documents/{language}",
        tenant_shared_template="{data_base_dir}/tenants/{tenant_slug}/shared/documents/{language}",
        user_processed_template="{data_base_dir}/tenants/{tenant_slug}/users/{user_id}/processed/{language}",
        tenant_processed_template="{data_base_dir}/tenants/{tenant_slug}/shared/processed/{language}",
        chromadb_path_template="{data_base_dir}/vectordb/{tenant_slug}",
        models_path_template="{models_base_dir}/{tenant_slug}/{language}",
        collection_name_template="{tenant_slug}_{scope}_{language}",
    )


# ================================
# INTEGRATION HELPERS
# ================================


def create_development_folder_manager():
    """Create folder manager configured for development/testing."""
    from .folder_manager import create_tenant_folder_manager

    config_provider, filesystem_provider, logger_provider = create_production_setup()
    return create_tenant_folder_manager(
        config_provider=config_provider,
        filesystem_provider=filesystem_provider,
        logger_provider=logger_provider,
    )


def create_test_folder_manager(
    config: Optional[FolderConfig] = None,
    existing_folders: Optional[Dict[str, bool]] = None,
    folder_stats: Optional[Dict[str, FolderStats]] = None,
    filesystem_failures: Optional[Dict[str, bool]] = None,
):
    """Create folder manager configured for testing."""
    from .folder_manager import create_tenant_folder_manager

    config_provider, filesystem_provider, logger_provider = create_mock_setup(
        config=config,
        existing_folders=existing_folders,
        folder_stats=folder_stats,
        filesystem_failures=filesystem_failures,
    )

    return create_tenant_folder_manager(
        config_provider=config_provider,
        filesystem_provider=filesystem_provider,
        logger_provider=logger_provider,
    ), (config_provider, filesystem_provider, logger_provider)
