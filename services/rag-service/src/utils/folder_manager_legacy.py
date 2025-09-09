"""
Multi-tenant folder management utilities.

Handles automatic creation and management of tenant/user/language folder structures.
Integrates with multi-tenant models and configuration system.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models.multitenant_models import (DocumentScope, Tenant,
                                         TenantUserContext, User)
from ..utils.config_loader import get_shared_config, get_storage_config

logger = logging.getLogger(__name__)


class TenantFolderManager:
    """Manages multi-tenant folder structure creation and organization."""

    def __init__(self, base_config: Optional[Dict] = None):
        """Initialize folder manager with configuration."""
        self.config = base_config or get_shared_config()
        self.storage_config = get_storage_config()

        # Base directories from config - fail fast if missing
        self.data_base = self.config["data_base_dir"]
        self.models_base = self.config["models_base_dir"]
        self.system_dir = self.config["system_dir"]

        # Path templates from config - fail fast if missing
        self.templates = {
            "tenant_root": self.config["tenant_root_template"],
            "user_documents": self.config["user_documents_template"],
            "tenant_shared": self.config["tenant_shared_template"],
            "user_processed": self.config["user_processed_template"],
            "tenant_processed": self.config["tenant_processed_template"],
            "tenant_chromadb": self.config["chromadb_path_template"],
            "tenant_models": self.config["models_path_template"],
            "collection_name": self.config["collection_name_template"],
        }

        logger.info(f"TenantFolderManager initialized with data base: {self.data_base}")

    def _render_template(self, template_name: str, **kwargs) -> str:
        """Render a path template with given parameters."""
        if template_name not in self.templates:
            raise KeyError(f"Template '{template_name}' not found in configuration")

        template = self.templates[template_name]

        # Add base directories to kwargs for template rendering
        render_kwargs = {
            "data_base_dir": self.data_base,
            "models_base_dir": self.models_base,
            "system_dir": self.system_dir,
            **kwargs,
        }

        return template.format(**render_kwargs)

    def get_tenant_folder_structure(
        self,
        tenant: Tenant,
        user: Optional[User] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Get complete folder structure for tenant/user/language combination."""
        tenant_slug = tenant.slug
        paths = {}

        # Base template parameters
        base_params = {
            "tenant_slug": tenant_slug,
            "language": language or "default",
        }

        if user:
            base_params["user_id"] = user.username

        # Tenant root using template
        tenant_root_str = self._render_template("tenant_root", **base_params)
        tenant_root = Path(tenant_root_str)
        paths["tenant_root"] = tenant_root

        # Core tenant folders (direct construction for non-templated paths)
        paths["tenant_cache"] = tenant_root / "cache"
        paths["tenant_exports"] = tenant_root / "exports"
        paths["tenant_logs"] = tenant_root / "logs"

        # ChromaDB path using template
        chromadb_str = self._render_template("tenant_chromadb", **base_params)
        paths["tenant_chromadb"] = Path(chromadb_str)

        # Language-specific tenant folders using templates
        if language:
            tenant_shared_str = self._render_template("tenant_shared", **base_params)
            paths["tenant_shared_documents_lang"] = Path(tenant_shared_str)

            tenant_processed_str = self._render_template(
                "tenant_processed", **base_params
            )
            paths["tenant_shared_processed_lang"] = Path(tenant_processed_str)

        # User-specific folders using templates
        if user:
            user_root = tenant_root / "users" / user.username
            paths["user_root"] = user_root
            paths["user_exports"] = user_root / "exports"
            paths["user_cache"] = user_root / "cache"
            paths["user_documents_raw"] = user_root / "documents" / "raw"

            # Language-specific user folders using templates
            if language:
                user_documents_str = self._render_template(
                    "user_documents", **base_params
                )
                paths["user_documents_lang"] = Path(user_documents_str)

                user_processed_str = self._render_template(
                    "user_processed", **base_params
                )
                paths["user_processed_lang"] = Path(user_processed_str)

        # Model folders using templates
        if language:
            models_str = self._render_template("tenant_models", **base_params)
            models_path = Path(models_str)
            paths["tenant_models_lang"] = models_path
            paths["tenant_models_embeddings"] = models_path / "embeddings"
            paths["tenant_models_generation"] = models_path / "generation"

        # Tenant models root (no language)
        models_root = Path(self.models_base) / tenant_slug
        paths["tenant_models"] = models_root
        paths["tenant_models_shared"] = models_root / "shared"

        return paths

    def create_tenant_folder_structure(
        self,
        tenant: Tenant,
        user: Optional[User] = None,
        languages: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """Create complete folder structure for tenant/user/languages.

        Returns:
            Tuple[bool, List[str]]: (success, list of created folders)
        """
        created_folders = []

        try:
            # Get supported languages if not provided
            if languages is None:
                languages = tenant.get_supported_languages()

            logger.info(
                f"Creating folder structure for tenant: {tenant.slug}, user: {user.username if user else None}, languages: {languages}"
            )

            # Create base structure for each language
            for language in languages:
                paths = self.get_tenant_folder_structure(tenant, user, language)

                # Create all paths
                for folder_type, folder_path in paths.items():
                    if self._create_folder(folder_path):
                        created_folders.append(str(folder_path))
                        logger.debug(f"Created {folder_type}: {folder_path}")

            # Create system folders if needed
            system_paths = [
                Path(self.system_dir) / "logs",
                Path(self.system_dir) / "backups",
                Path(self.system_dir) / "temp",
            ]

            for system_path in system_paths:
                if self._create_folder(system_path):
                    created_folders.append(str(system_path))

            logger.info(
                f"Successfully created {len(created_folders)} folders for tenant structure"
            )
            return True, created_folders

        except Exception as e:
            logger.error(f"Failed to create tenant folder structure: {e}")
            return False, created_folders

    def get_tenant_document_path(
        self,
        context: TenantUserContext,
        language: str,
        scope: DocumentScope,
        create_if_missing: bool = True,
    ) -> Path:
        """Get document storage path for specific tenant/user/language/scope."""
        paths = self.get_tenant_folder_structure(context.tenant, context.user, language)

        if scope == DocumentScope.USER:
            doc_path = paths.get("user_documents_lang")
        else:  # DocumentScope.TENANT
            doc_path = paths.get("tenant_shared_documents_lang")

        if create_if_missing and doc_path:
            self._create_folder(doc_path)

        return doc_path

    def get_tenant_processed_path(
        self,
        context: TenantUserContext,
        language: str,
        scope: DocumentScope,
        create_if_missing: bool = True,
    ) -> Path:
        """Get processed data storage path for specific tenant/user/language/scope."""
        paths = self.get_tenant_folder_structure(context.tenant, context.user, language)

        if scope == DocumentScope.USER:
            processed_path = paths.get("user_processed_lang")
        else:  # DocumentScope.TENANT
            processed_path = paths.get("tenant_shared_processed_lang")

        if create_if_missing and processed_path:
            self._create_folder(processed_path)

        return processed_path

    def get_tenant_chromadb_path(
        self, tenant: Tenant, create_if_missing: bool = True
    ) -> Path:
        """Get ChromaDB storage path for tenant."""
        chromadb_str = self._render_template("tenant_chromadb", tenant_slug=tenant.slug)
        chromadb_path = Path(chromadb_str)

        if create_if_missing:
            self._create_folder(chromadb_path)

        return chromadb_path

    def get_tenant_models_path(
        self,
        tenant: Tenant,
        language: str,
        model_type: str = "embeddings",
        create_if_missing: bool = True,
    ) -> Path:
        """Get model storage path for tenant/language."""
        models_str = self._render_template(
            "tenant_models", tenant_slug=tenant.slug, language=language
        )
        models_path = Path(models_str) / model_type

        if create_if_missing:
            self._create_folder(models_path)

        return models_path

    def get_collection_storage_paths(
        self, context: TenantUserContext, language: str
    ) -> Dict[str, Path]:
        """Get all ChromaDB collection storage paths for a tenant/user/language."""
        chromadb_base = self.get_tenant_chromadb_path(context.tenant)

        # Use template for collection names
        user_collection = self._render_template(
            "collection_name",
            tenant_slug=context.tenant.slug,
            scope="user",
            language=language,
        )
        tenant_collection = self._render_template(
            "collection_name",
            tenant_slug=context.tenant.slug,
            scope="tenant",
            language=language,
        )

        return {
            "user_collection_name": user_collection,
            "tenant_collection_name": tenant_collection,
            "user_collection_path": chromadb_base / user_collection,
            "tenant_collection_path": chromadb_base / tenant_collection,
            "base_path": chromadb_base,
        }

    def ensure_context_folders(self, context: TenantUserContext, language: str) -> bool:
        """Ensure all necessary folders exist for a tenant/user/language context."""
        try:
            # Create tenant and user folder structure
            success, folders = self.create_tenant_folder_structure(
                context.tenant, context.user, [language]
            )

            if success:
                logger.info(
                    f"Ensured folder structure for {context.tenant.slug}/{context.user.username}/{language}"
                )
                return True
            else:
                logger.error(f"Failed to ensure folder structure for context")
                return False

        except Exception as e:
            logger.error(f"Error ensuring context folders: {e}")
            return False

    def cleanup_tenant_folders(self, tenant: Tenant, confirm: bool = False) -> bool:
        """Clean up all folders for a tenant. USE WITH CAUTION."""
        if not confirm:
            logger.warning("Cleanup requires explicit confirmation. Set confirm=True")
            return False

        try:
            import shutil

            tenant_root_str = self._render_template(
                "tenant_root", tenant_slug=tenant.slug
            )
            tenant_root = Path(tenant_root_str)
            models_root = Path(self.models_base) / tenant.slug

            if tenant_root.exists():
                shutil.rmtree(tenant_root)
                logger.info(f"Cleaned up tenant data: {tenant_root}")

            if models_root.exists():
                shutil.rmtree(models_root)
                logger.info(f"Cleaned up tenant models: {models_root}")

            return True

        except Exception as e:
            logger.error(f"Failed to cleanup tenant folders: {e}")
            return False

    def get_folder_usage_stats(self, tenant: Tenant) -> Dict[str, Dict[str, int]]:
        """Get storage usage statistics for tenant."""
        stats = {
            "documents": {"count": 0, "size_bytes": 0},
            "processed": {"count": 0, "size_bytes": 0},
            "models": {"count": 0, "size_bytes": 0},
            "chromadb": {"count": 0, "size_bytes": 0},
        }

        try:
            tenant_root_str = self._render_template(
                "tenant_root", tenant_slug=tenant.slug
            )
            tenant_root = Path(tenant_root_str)
            models_root = Path(self.models_base) / tenant.slug

            # Count documents using template paths
            if tenant_root.exists():
                chromadb_str = self._render_template(
                    "tenant_chromadb", tenant_slug=tenant.slug
                )
                for path_type, path in [
                    ("documents", tenant_root / "users"),
                    ("processed", tenant_root / "shared"),
                    ("chromadb", Path(chromadb_str)),
                ]:
                    if path.exists():
                        stats[path_type] = self._get_folder_stats(path)

            # Count models
            if models_root.exists():
                stats["models"] = self._get_folder_stats(models_root)

        except Exception as e:
            logger.error(f"Error getting folder stats: {e}")

        return stats

    def _create_folder(self, folder_path: Path) -> bool:
        """Create a folder if it doesn't exist."""
        try:
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created folder: {folder_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to create folder {folder_path}: {e}")
            return False

    def _get_folder_stats(self, folder_path: Path) -> Dict[str, int]:
        """Get file count and size for a folder."""
        stats = {"count": 0, "size_bytes": 0}

        try:
            if folder_path.exists():
                for file_path in folder_path.rglob("*"):
                    if file_path.is_file():
                        stats["count"] += 1
                        stats["size_bytes"] += file_path.stat().st_size
        except Exception as e:
            logger.error(f"Error getting stats for {folder_path}: {e}")

        return stats


def create_development_structure() -> bool:
    """Create default development tenant/user structure for testing."""
    from ..models.multitenant_models import (DEFAULT_DEVELOPMENT_TENANT,
                                             DEFAULT_DEVELOPMENT_USER)

    folder_manager = TenantFolderManager()

    # Create structure for development tenant with default user
    success, folders = folder_manager.create_tenant_folder_structure(
        tenant=DEFAULT_DEVELOPMENT_TENANT,
        user=DEFAULT_DEVELOPMENT_USER,
        languages=["hr", "en", "multilingual"],
    )

    if success:
        logger.info(f"Created development structure with {len(folders)} folders")
        logger.info("Ready for user-level multi-tenant testing")
    else:
        logger.error("Failed to create development structure")

    return success


def ensure_tenant_context_ready(context: TenantUserContext, language: str) -> bool:
    """Ensure all folders are ready for a specific tenant/user/language context."""
    folder_manager = TenantFolderManager()
    return folder_manager.ensure_context_folders(context, language)


# Convenience functions for common operations
def get_user_document_folder(context: TenantUserContext, language: str) -> Path:
    """Get user's document folder path."""
    folder_manager = TenantFolderManager()
    return folder_manager.get_tenant_document_path(
        context, language, DocumentScope.USER
    )


def get_tenant_document_folder(context: TenantUserContext, language: str) -> Path:
    """Get tenant's shared document folder path."""
    folder_manager = TenantFolderManager()
    return folder_manager.get_tenant_document_path(
        context, language, DocumentScope.TENANT
    )


def get_user_chromadb_collections(
    context: TenantUserContext, language: str
) -> Dict[str, str]:
    """Get ChromaDB collection names for user context."""
    return {
        "user_collection": context.get_user_collection_name(language),
        "tenant_collection": context.get_tenant_collection_name(language),
        "search_collections": context.get_search_collections(language),
    }
