"""
Multi-tenant folder management utilities with dependency injection.
Testable version with pure functions and dependency injection architecture.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..models.multitenant_models import DocumentScope, Tenant, TenantUserContext, User
from .logging_factory import get_system_logger, log_component_end, log_component_start, log_error_context

# ================================
# DATA CLASSES & CONFIGURATION
# ================================


@dataclass
class FolderConfig:
    """Configuration for folder management operations."""

    data_base_dir: str
    models_base_dir: str
    system_dir: str
    tenant_root_template: str
    user_documents_template: str
    tenant_shared_template: str
    user_processed_template: str
    tenant_processed_template: str
    chromadb_path_template: str
    models_path_template: str
    collection_name_template: str


@dataclass
class FolderStats:
    """Statistics for folder usage."""

    count: int = 0
    size_bytes: int = 0


@dataclass
class TenantStats:
    """Complete statistics for a tenant."""

    documents: FolderStats
    processed: FolderStats
    models: FolderStats
    chromadb: FolderStats


@dataclass
class FolderPaths:
    """Complete folder structure for a tenant/user/language combination."""

    tenant_root: Path
    tenant_cache: Path | None = None
    tenant_exports: Path | None = None
    tenant_logs: Path | None = None
    tenant_chromadb: Path | None = None
    tenant_shared_documents_lang: Path | None = None
    tenant_shared_processed_lang: Path | None = None
    user_root: Path | None = None
    user_exports: Path | None = None
    user_cache: Path | None = None
    user_documents_raw: Path | None = None
    user_documents_lang: Path | None = None
    user_processed_lang: Path | None = None
    tenant_models_lang: Path | None = None
    tenant_models_embeddings: Path | None = None
    tenant_models_generation: Path | None = None
    tenant_models: Path | None = None
    tenant_models_shared: Path | None = None


@dataclass
class CollectionPaths:
    """ChromaDB collection paths and names."""

    user_collection_name: str
    tenant_collection_name: str
    user_collection_path: Path
    tenant_collection_path: Path
    base_path: Path


# ================================
# DEPENDENCY INJECTION PROTOCOLS
# ================================


class FileSystemProvider(Protocol):
    """Protocol for filesystem operations."""

    def create_folder(self, folder_path: Path) -> bool:
        """Create a folder if it doesn't exist."""
        ...

    def folder_exists(self, folder_path: Path) -> bool:
        """Check if folder exists."""
        ...

    def remove_folder(self, folder_path: Path) -> bool:
        """Remove folder and all contents."""
        ...

    def get_folder_stats(self, folder_path: Path) -> FolderStats:
        """Get file count and size statistics for folder."""
        ...


class ConfigProvider(Protocol):
    """Protocol for configuration access."""

    def get_folder_config(self) -> FolderConfig:
        """Get folder configuration."""
        ...


class LoggerProvider(Protocol):
    """Protocol for logging operations."""

    def info(self, message: str) -> None:
        """Log info message."""
        ...

    def debug(self, message: str) -> None:
        """Log debug message."""
        ...

    def warning(self, message: str) -> None:
        """Log warning message."""
        ...

    def error(self, message: str) -> None:
        """Log error message."""
        ...


# ================================
# PURE BUSINESS LOGIC FUNCTIONS
# ================================


def render_path_template(template: str, **kwargs) -> str:
    """Pure function to render a path template with given parameters."""
    logger = get_system_logger()
    logger.trace("folder_template", "render_path_template", f"Rendering template: {template}", metadata=kwargs)

    result = template.format(**kwargs)
    logger.trace("folder_template", "render_path_template", f"Rendered result: {result}")
    return result


def build_template_params(
    tenant: Tenant, user: User | None = None, language: str | None = None, config: FolderConfig | None = None
) -> dict[str, str]:
    """Pure function to build template parameters for path rendering."""
    logger = get_system_logger()
    logger.trace(
        "folder_template",
        "build_template_params",
        "Building template parameters",
        metadata={"tenant_slug": tenant.slug, "user": user.username if user else None, "language": language},
    )

    params = {"tenant_slug": tenant.slug, "language": language or "default"}

    if user:
        logger.trace("folder_template", "build_template_params", f"Adding user parameter: {user.username}")
        params["user_id"] = user.username

    if config:
        logger.trace("folder_template", "build_template_params", "Adding config-based parameters")
        params.update(
            {
                "data_base_dir": config.data_base_dir,
                "models_base_dir": config.models_base_dir,
                "system_dir": config.system_dir,
            }
        )

    logger.trace("folder_template", "build_template_params", f"Built parameters: {params}")
    return params


def calculate_folder_structure(
    tenant: Tenant, user: User | None, language: str | None, config: FolderConfig
) -> FolderPaths:
    """Pure function to calculate complete folder structure."""
    logger = get_system_logger()
    log_component_start(
        "folder_calculator",
        "calculate_folder_structure",
        tenant_slug=tenant.slug,
        user=user.username if user else None,
        language=language,
    )

    params = build_template_params(tenant, user, language, config)

    # Render tenant root
    logger.debug("folder_calculator", "calculate_folder_structure", "Rendering tenant root path")
    tenant_root_str = render_path_template(config.tenant_root_template, **params)
    tenant_root = Path(tenant_root_str)

    # Build folder paths structure
    paths = FolderPaths(tenant_root=tenant_root)
    logger.debug("folder_calculator", "calculate_folder_structure", f"Tenant root: {tenant_root}")

    # Core tenant folders (direct construction)
    logger.trace("folder_calculator", "calculate_folder_structure", "Creating core tenant folders")
    paths.tenant_cache = tenant_root / "cache"
    paths.tenant_exports = tenant_root / "exports"
    paths.tenant_logs = tenant_root / "logs"

    # ChromaDB path using template
    logger.debug("folder_calculator", "calculate_folder_structure", "Calculating ChromaDB path")
    chromadb_str = render_path_template(config.chromadb_path_template, **params)
    paths.tenant_chromadb = Path(chromadb_str)

    # Language-specific tenant folders
    if language:
        logger.debug(
            "folder_calculator", "calculate_folder_structure", f"Creating language-specific folders for {language}"
        )
        tenant_shared_str = render_path_template(config.tenant_shared_template, **params)
        paths.tenant_shared_documents_lang = Path(tenant_shared_str)

        tenant_processed_str = render_path_template(config.tenant_processed_template, **params)
        paths.tenant_shared_processed_lang = Path(tenant_processed_str)

    # User-specific folders
    if user:
        logger.debug(
            "folder_calculator", "calculate_folder_structure", f"Creating user-specific folders for {user.username}"
        )
        user_root = tenant_root / "users" / user.username
        paths.user_root = user_root
        paths.user_exports = user_root / "exports"
        paths.user_cache = user_root / "cache"
        paths.user_documents_raw = user_root / "documents" / "raw"

        # Language-specific user folders
        if language:
            logger.trace(
                "folder_calculator", "calculate_folder_structure", f"Adding user language folders for {language}"
            )
            user_documents_str = render_path_template(config.user_documents_template, **params)
            paths.user_documents_lang = Path(user_documents_str)

            user_processed_str = render_path_template(config.user_processed_template, **params)
            paths.user_processed_lang = Path(user_processed_str)

    # Model folders
    if language:
        logger.debug("folder_calculator", "calculate_folder_structure", f"Creating model folders for {language}")
        models_str = render_path_template(config.models_path_template, **params)
        models_path = Path(models_str)
        paths.tenant_models_lang = models_path
        paths.tenant_models_embeddings = models_path / "embeddings"
        paths.tenant_models_generation = models_path / "generation"

    # Tenant models root (no language)
    logger.trace("folder_calculator", "calculate_folder_structure", "Creating tenant models root")
    models_root = Path(config.models_base_dir) / tenant.slug
    paths.tenant_models = models_root
    paths.tenant_models_shared = models_root / "shared"

    folder_count = len([p for p in get_all_folder_paths(paths) if p is not None])
    log_component_end("folder_calculator", "calculate_folder_structure", f"Calculated {folder_count} folder paths")
    return paths


def calculate_document_path(
    context: TenantUserContext, language: str, scope: DocumentScope, config: FolderConfig
) -> Path:
    """Pure function to calculate document storage path."""
    logger = get_system_logger()
    logger.debug(
        "folder_calculator",
        "calculate_document_path",
        f"Calculating document path for scope: {scope.value}",
        metadata={"tenant": context.tenant.slug, "user": context.user.username, "language": language},
    )

    paths = calculate_folder_structure(context.tenant, context.user, language, config)

    if scope == DocumentScope.USER:
        assert paths.user_documents_lang is not None, "User documents path not configured"
        logger.trace(
            "folder_calculator", "calculate_document_path", f"User documents path: {paths.user_documents_lang}"
        )
        return paths.user_documents_lang
    else:  # DocumentScope.TENANT
        assert paths.tenant_shared_documents_lang is not None, "Tenant documents path not configured"
        logger.trace(
            "folder_calculator",
            "calculate_document_path",
            f"Tenant documents path: {paths.tenant_shared_documents_lang}",
        )
        return paths.tenant_shared_documents_lang


def calculate_processed_path(
    context: TenantUserContext, language: str, scope: DocumentScope, config: FolderConfig
) -> Path:
    """Pure function to calculate processed data storage path."""
    paths = calculate_folder_structure(context.tenant, context.user, language, config)

    if scope == DocumentScope.USER:
        assert paths.user_processed_lang is not None, "User processed path not configured"
        return paths.user_processed_lang
    else:  # DocumentScope.TENANT
        assert paths.tenant_shared_processed_lang is not None, "Tenant processed path not configured"
        return paths.tenant_shared_processed_lang


def calculate_chromadb_path(tenant: Tenant, config: FolderConfig) -> Path:
    """Pure function to calculate ChromaDB storage path."""
    params = build_template_params(tenant, config=config)
    chromadb_str = render_path_template(config.chromadb_path_template, **params)
    return Path(chromadb_str)


def calculate_models_path(tenant: Tenant, language: str, model_type: str, config: FolderConfig) -> Path:
    """Pure function to calculate model storage path."""
    params = build_template_params(tenant, language=language, config=config)
    models_str = render_path_template(config.models_path_template, **params)
    return Path(models_str) / model_type


def calculate_collection_paths(context: TenantUserContext, language: str, config: FolderConfig) -> CollectionPaths:
    """Pure function to calculate ChromaDB collection paths and names."""
    chromadb_base = calculate_chromadb_path(context.tenant, config)
    params = build_template_params(context.tenant, config=config)

    # Calculate collection names
    user_params = {**params, "scope": "user", "language": language}
    tenant_params = {**params, "scope": "tenant", "language": language}

    user_collection = render_path_template(config.collection_name_template, **user_params)
    tenant_collection = render_path_template(config.collection_name_template, **tenant_params)

    return CollectionPaths(
        user_collection_name=user_collection,
        tenant_collection_name=tenant_collection,
        user_collection_path=chromadb_base / user_collection,
        tenant_collection_path=chromadb_base / tenant_collection,
        base_path=chromadb_base,
    )


def get_system_paths(config: FolderConfig) -> list[Path]:
    """Pure function to get system folder paths."""
    system_dir = Path(config.system_dir)
    return [system_dir / "logs", system_dir / "backups", system_dir / "temp"]


def get_all_folder_paths(paths: FolderPaths) -> list[Path]:
    """Pure function to extract all non-None paths from FolderPaths structure."""
    all_paths = []
    for _field_name, field_value in paths.__dict__.items():
        if field_value is not None:
            all_paths.append(field_value)
    return all_paths


def calculate_tenant_cleanup_paths(tenant: Tenant, config: FolderConfig) -> list[Path]:
    """Pure function to calculate paths that need to be cleaned up for a tenant."""
    params = build_template_params(tenant, config=config)
    tenant_root_str = render_path_template(config.tenant_root_template, **params)
    tenant_root = Path(tenant_root_str)
    models_root = Path(config.models_base_dir) / tenant.slug

    return [tenant_root, models_root]


def calculate_usage_stats_paths(tenant: Tenant, config: FolderConfig) -> dict[str, Path]:
    """Pure function to calculate paths for usage statistics."""
    params = build_template_params(tenant, config=config)
    tenant_root_str = render_path_template(config.tenant_root_template, **params)
    tenant_root = Path(tenant_root_str)
    models_root = Path(config.models_base_dir) / tenant.slug
    chromadb_str = render_path_template(config.chromadb_path_template, **params)

    return {
        "documents": tenant_root / "users",
        "processed": tenant_root / "shared",
        "chromadb": Path(chromadb_str),
        "models": models_root,
    }


# ================================
# DEPENDENCY INJECTION ORCHESTRATION
# ================================


class _TenantFolderManager:
    """Testable tenant folder manager with dependency injection."""

    def __init__(
        self,
        config_provider: ConfigProvider,
        filesystem_provider: FileSystemProvider,
        logger_provider: LoggerProvider | None = None,
    ):
        """Initialize with injected dependencies."""
        logger = get_system_logger()
        log_component_start("folder_manager", "init", has_logger=logger_provider is not None)

        self._config_provider = config_provider
        self._filesystem_provider = filesystem_provider
        self._logger = logger_provider
        self._config: FolderConfig | None = None

        logger.debug("folder_manager", "init", "TenantFolderManager initialized with dependency injection")
        log_component_end("folder_manager", "init", "TenantFolderManager ready for operations")

    def _get_config(self) -> FolderConfig:
        """Get configuration with caching."""
        if self._config is None:
            self._config = self._config_provider.get_folder_config()
        return self._config

    def _log_info(self, message: str) -> None:
        """Log info message if logger available."""
        if self._logger:
            self._logger.info(message)

    def _log_debug(self, message: str) -> None:
        """Log debug message if logger available."""
        if self._logger:
            self._logger.debug(message)

    def _log_warning(self, message: str) -> None:
        """Log warning message if logger available."""
        if self._logger:
            self._logger.warning(message)

    def _log_error(self, message: str) -> None:
        """Log error message if logger available."""
        if self._logger:
            self._logger.error(message)

    def get_tenant_folder_structure(
        self, tenant: Tenant, user: User | None = None, language: str | None = None
    ) -> FolderPaths:
        """Get complete folder structure for tenant/user/language combination."""
        config = self._get_config()
        return calculate_folder_structure(tenant, user, language, config)

    def create_tenant_folder_structure(
        self, tenant: Tenant, user: User | None = None, languages: list[str] | None = None
    ) -> tuple[bool, list[str]]:
        """Create complete folder structure for tenant/user/languages."""
        logger = get_system_logger()
        log_component_start(
            "folder_manager",
            "create_tenant_folder_structure",
            tenant_slug=tenant.slug,
            user=user.username if user else None,
            languages=languages,
        )

        created_folders = []

        try:
            config = self._get_config()

            # Get supported languages if not provided
            if languages is None:
                languages = tenant.get_supported_languages()
                logger.debug(
                    "folder_manager", "create_tenant_folder_structure", f"Using tenant default languages: {languages}"
                )

            logger.info(
                "folder_manager",
                "create_tenant_folder_structure",
                f"Creating folder structure for tenant: {tenant.slug}, user: {user.username if user else None}, languages: {languages}",
            )

            # Create base structure for each language
            for language in languages:
                logger.debug("folder_manager", "create_tenant_folder_structure", f"Processing language: {language}")
                paths = calculate_folder_structure(tenant, user, language, config)

                # Create all paths
                for folder_path in get_all_folder_paths(paths):
                    if self._filesystem_provider.create_folder(folder_path):
                        created_folders.append(str(folder_path))
                        logger.trace(
                            "folder_manager", "create_tenant_folder_structure", f"Created folder: {folder_path}"
                        )

            # Create system folders if needed
            logger.debug("folder_manager", "create_tenant_folder_structure", "Creating system folders")
            system_paths = get_system_paths(config)
            for system_path in system_paths:
                if self._filesystem_provider.create_folder(system_path):
                    created_folders.append(str(system_path))
                    logger.trace(
                        "folder_manager", "create_tenant_folder_structure", f"Created system folder: {system_path}"
                    )

            logger.info(
                "folder_manager",
                "create_tenant_folder_structure",
                f"Successfully created {len(created_folders)} folders for tenant structure",
            )
            log_component_end(
                "folder_manager",
                "create_tenant_folder_structure",
                f"Created {len(created_folders)} folders successfully",
            )
            return True, created_folders

        except Exception as e:
            log_error_context(
                "folder_manager",
                "create_tenant_folder_structure",
                e,
                {"tenant_slug": tenant.slug, "user": user.username if user else None, "languages": languages},
            )
            raise

    def get_tenant_document_path(
        self, context: TenantUserContext, language: str, scope: DocumentScope, create_if_missing: bool = True
    ) -> Path:
        """Get document storage path for specific tenant/user/language/scope."""
        config = self._get_config()
        doc_path = calculate_document_path(context, language, scope, config)

        if create_if_missing and doc_path:
            self._filesystem_provider.create_folder(doc_path)

        return doc_path

    def get_tenant_processed_path(
        self, context: TenantUserContext, language: str, scope: DocumentScope, create_if_missing: bool = True
    ) -> Path:
        """Get processed data storage path for specific tenant/user/language/scope."""
        config = self._get_config()
        processed_path = calculate_processed_path(context, language, scope, config)

        if create_if_missing and processed_path:
            self._filesystem_provider.create_folder(processed_path)

        return processed_path

    def get_tenant_chromadb_path(self, tenant: Tenant, create_if_missing: bool = True) -> Path:
        """Get ChromaDB storage path for tenant."""
        config = self._get_config()
        chromadb_path = calculate_chromadb_path(tenant, config)

        if create_if_missing:
            self._filesystem_provider.create_folder(chromadb_path)

        return chromadb_path

    def get_tenant_models_path(
        self, tenant: Tenant, language: str, model_type: str = "embeddings", create_if_missing: bool = True
    ) -> Path:
        """Get model storage path for tenant/language."""
        config = self._get_config()
        models_path = calculate_models_path(tenant, language, model_type, config)

        if create_if_missing:
            self._filesystem_provider.create_folder(models_path)

        return models_path

    def get_collection_storage_paths(self, context: TenantUserContext, language: str) -> CollectionPaths:
        """Get all ChromaDB collection storage paths for a tenant/user/language."""
        config = self._get_config()
        return calculate_collection_paths(context, language, config)

    def ensure_context_folders(self, context: TenantUserContext, language: str) -> bool:
        """Ensure all necessary folders exist for a tenant/user/language context."""
        try:
            # Create tenant and user folder structure
            success, folders = self.create_tenant_folder_structure(context.tenant, context.user, [language])

            if success:
                self._log_info(f"Ensured folder structure for {context.tenant.slug}/{context.user.username}/{language}")
                return True
            else:
                self._log_error("Failed to ensure folder structure for context")
                return False

        except Exception as e:
            self._log_error(f"Error ensuring context folders: {e}")
            raise

    def cleanup_tenant_folders(self, tenant: Tenant, confirm: bool = False) -> bool:
        """Clean up all folders for a tenant. USE WITH CAUTION."""
        logger = get_system_logger()
        log_component_start("folder_manager", "cleanup_tenant_folders", tenant_slug=tenant.slug, confirm=confirm)

        if not confirm:
            logger.warning(
                "folder_manager", "cleanup_tenant_folders", "Cleanup requires explicit confirmation. Set confirm=True"
            )
            return False

        try:
            config = self._get_config()
            cleanup_paths = calculate_tenant_cleanup_paths(tenant, config)
            logger.info(
                "folder_manager",
                "cleanup_tenant_folders",
                f"Cleaning up {len(cleanup_paths)} paths for tenant {tenant.slug}",
            )

            for path in cleanup_paths:
                logger.debug("folder_manager", "cleanup_tenant_folders", f"Checking path: {path}")
                if self._filesystem_provider.folder_exists(path):
                    logger.warning("folder_manager", "cleanup_tenant_folders", f"Removing folder: {path}")
                    if self._filesystem_provider.remove_folder(path):
                        logger.info(
                            "folder_manager", "cleanup_tenant_folders", f"Successfully cleaned up folder: {path}"
                        )
                    else:
                        logger.error("folder_manager", "cleanup_tenant_folders", f"Failed to clean up folder: {path}")
                        return False
                else:
                    logger.trace("folder_manager", "cleanup_tenant_folders", f"Path does not exist, skipping: {path}")

            log_component_end(
                "folder_manager", "cleanup_tenant_folders", f"Tenant {tenant.slug} cleanup completed successfully"
            )
            return True

        except Exception as e:
            log_error_context(
                "folder_manager", "cleanup_tenant_folders", e, {"tenant_slug": tenant.slug, "confirm": confirm}
            )
            raise

    def get_folder_usage_stats(self, tenant: Tenant) -> TenantStats:
        """Get storage usage statistics for tenant."""
        logger = get_system_logger()
        log_component_start("folder_manager", "get_folder_usage_stats", tenant_slug=tenant.slug)

        try:
            config = self._get_config()
            stats_paths = calculate_usage_stats_paths(tenant, config)
            logger.debug(
                "folder_manager",
                "get_folder_usage_stats",
                f"Analyzing {len(stats_paths)} path types for tenant {tenant.slug}",
            )

            stats = TenantStats(
                documents=FolderStats(), processed=FolderStats(), models=FolderStats(), chromadb=FolderStats()
            )

            # Get stats for each path type
            for stat_type, path in stats_paths.items():
                logger.trace("folder_manager", "get_folder_usage_stats", f"Checking {stat_type} path: {path}")
                if self._filesystem_provider.folder_exists(path):
                    folder_stats = self._filesystem_provider.get_folder_stats(path)
                    setattr(stats, stat_type, folder_stats)
                    logger.debug(
                        "folder_manager",
                        "get_folder_usage_stats",
                        f"{stat_type}: {folder_stats.count} files, {folder_stats.size_bytes} bytes",
                    )
                else:
                    logger.trace("folder_manager", "get_folder_usage_stats", f"{stat_type} path does not exist: {path}")

            total_files = stats.documents.count + stats.processed.count + stats.models.count + stats.chromadb.count
            total_size = (
                stats.documents.size_bytes
                + stats.processed.size_bytes
                + stats.models.size_bytes
                + stats.chromadb.size_bytes
            )
            log_component_end(
                "folder_manager", "get_folder_usage_stats", f"Stats collected: {total_files} files, {total_size} bytes"
            )
            return stats

        except Exception as e:
            log_error_context("folder_manager", "get_folder_usage_stats", e, {"tenant_slug": tenant.slug})
            raise


# ================================
# CONVENIENCE FACTORY FUNCTIONS
# ================================


def create_tenant_folder_manager(
    config_provider: ConfigProvider,
    filesystem_provider: FileSystemProvider,
    logger_provider: LoggerProvider | None = None,
) -> _TenantFolderManager:
    """Factory function to create configured TenantFolderManager."""
    return _TenantFolderManager(
        config_provider=config_provider, filesystem_provider=filesystem_provider, logger_provider=logger_provider
    )


# ================================
# PUBLIC INTERFACE
# ================================


def TenantFolderManager(
    base_config: dict | None = None,
    config_provider: ConfigProvider | None = None,
    filesystem_provider: FileSystemProvider | None = None,
    logger_provider: LoggerProvider | None = None,
):
    """
    Create a tenant folder manager with dependency injection.

    Args:
        base_config: Base configuration (optional fallback)
        config_provider: Configuration provider for folder settings
        filesystem_provider: Filesystem provider for operations
        logger_provider: Logger provider for debugging

    Returns:
        Configured _TenantFolderManager instance
    """
    if not config_provider or not filesystem_provider:
        from .folder_manager_providers import create_production_setup

        (config_provider, filesystem_provider, logger_provider) = create_production_setup()

    # Ensure providers are not None after potential defaults
    assert config_provider is not None, "ConfigProvider must not be None"
    assert filesystem_provider is not None, "FileSystemProvider must not be None"

    return _TenantFolderManager(
        config_provider=config_provider, filesystem_provider=filesystem_provider, logger_provider=logger_provider
    )
