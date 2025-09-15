"""
Tests for folder manager module.
Comprehensive testing of pure functions and dependency injection architecture.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock

from src.utils.folder_manager import (
    # Data classes
    FolderConfig,
    FolderStats,
    TenantStats,
    FolderPaths,
    CollectionPaths,
    # Protocols
    FileSystemProvider,
    ConfigProvider,
    LoggerProvider,
    # Pure functions
    render_path_template,
    build_template_params,
    calculate_folder_structure,
    calculate_document_path,
    calculate_processed_path,
    calculate_chromadb_path,
    calculate_models_path,
    calculate_collection_paths,
    get_system_paths,
    get_all_folder_paths,
    calculate_tenant_cleanup_paths,
    calculate_usage_stats_paths,
    # Main class and factory
    _TenantFolderManager,
    create_tenant_folder_manager,
    TenantFolderManager,
)
from src.models.multitenant_models import DocumentScope, Tenant, TenantUserContext, User


class TestFolderConfig:
    """Test FolderConfig data class."""

    def test_folder_config_creation(self):
        """Test creating folder configuration."""
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="{data_base_dir}/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}_documents"
        )

        assert config.data_base_dir == "/app/data"
        assert config.models_base_dir == "/app/models"
        assert config.system_dir == "/app/system"
        assert "{tenant_slug}" in config.tenant_root_template


class TestFolderStats:
    """Test FolderStats data class."""

    def test_folder_stats_creation(self):
        """Test creating folder statistics."""
        stats = FolderStats(count=10, size_bytes=1024)

        assert stats.count == 10
        assert stats.size_bytes == 1024

    def test_folder_stats_defaults(self):
        """Test folder statistics with defaults."""
        stats = FolderStats()

        assert stats.count == 0
        assert stats.size_bytes == 0


class TestTenantStats:
    """Test TenantStats data class."""

    def test_tenant_stats_creation(self):
        """Test creating tenant statistics."""
        stats = TenantStats(
            documents=FolderStats(5, 512),
            processed=FolderStats(3, 256),
            models=FolderStats(2, 128),
            chromadb=FolderStats(1, 64)
        )

        assert stats.documents.count == 5
        assert stats.processed.count == 3
        assert stats.models.count == 2
        assert stats.chromadb.count == 1


class TestFolderPaths:
    """Test FolderPaths data class."""

    def test_folder_paths_creation(self):
        """Test creating folder paths structure."""
        tenant_root = Path("/app/data/tenant1")
        paths = FolderPaths(
            tenant_root=tenant_root,
            tenant_cache=tenant_root / "cache",
            user_root=tenant_root / "users" / "user1"
        )

        assert paths.tenant_root == tenant_root
        assert paths.tenant_cache == tenant_root / "cache"
        assert paths.user_root == tenant_root / "users" / "user1"

    def test_folder_paths_optional_fields(self):
        """Test folder paths with only required fields."""
        tenant_root = Path("/app/data/tenant1")
        paths = FolderPaths(tenant_root=tenant_root)

        assert paths.tenant_root == tenant_root
        assert paths.tenant_cache is None
        assert paths.user_root is None


class TestCollectionPaths:
    """Test CollectionPaths data class."""

    def test_collection_paths_creation(self):
        """Test creating collection paths."""
        base_path = Path("/app/data/tenant1/chromadb")
        paths = CollectionPaths(
            user_collection_name="tenant1_user_hr_documents",
            tenant_collection_name="tenant1_tenant_hr_documents",
            user_collection_path=base_path / "tenant1_user_hr_documents",
            tenant_collection_path=base_path / "tenant1_tenant_hr_documents",
            base_path=base_path
        )

        assert paths.user_collection_name == "tenant1_user_hr_documents"
        assert paths.tenant_collection_name == "tenant1_tenant_hr_documents"
        assert paths.base_path == base_path


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_render_path_template_basic(self):
        """Test basic path template rendering."""
        template = "/data/{tenant_slug}/users/{user_id}"
        result = render_path_template(template, tenant_slug="tenant1", user_id="user1")

        assert result == "/data/tenant1/users/user1"

    def test_render_path_template_missing_param(self):
        """Test path template rendering with missing parameter."""
        template = "/data/{tenant_slug}/{missing}"

        with pytest.raises(KeyError):
            render_path_template(template, tenant_slug="tenant1")

    def test_build_template_params_basic(self):
        """Test building template parameters."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        user = User(id="1", tenant_id="1", username="user1", email="user1@test.com")

        params = build_template_params(tenant, user, "hr")

        assert params["tenant_slug"] == "tenant1"
        assert params["user_id"] == "user1"
        assert params["language"] == "hr"

    def test_build_template_params_no_user(self):
        """Test building template parameters without user."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")

        params = build_template_params(tenant, None, "en")

        assert params["tenant_slug"] == "tenant1"
        assert "user_id" not in params
        assert params["language"] == "en"

    def test_build_template_params_no_language(self):
        """Test building template parameters without language."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")

        params = build_template_params(tenant)

        assert params["tenant_slug"] == "tenant1"
        assert params["language"] == "default"

    def test_build_template_params_with_config(self):
        """Test building template parameters with config."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="",
            user_documents_template="",
            tenant_shared_template="",
            user_processed_template="",
            tenant_processed_template="",
            chromadb_path_template="",
            models_path_template="",
            collection_name_template=""
        )

        params = build_template_params(tenant, config=config)

        assert params["data_base_dir"] == "/app/data"
        assert params["models_base_dir"] == "/app/models"
        assert params["system_dir"] == "/app/system"

    def test_calculate_folder_structure_complete(self):
        """Test calculating complete folder structure."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        user = User(id="1", tenant_id="1", username="user1", email="user1@test.com")
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="{data_base_dir}/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}_documents"
        )

        paths = calculate_folder_structure(tenant, user, "hr", config)

        assert paths.tenant_root == Path("/app/data/tenant1")
        assert paths.tenant_cache == Path("/app/data/tenant1/cache")
        assert paths.tenant_exports == Path("/app/data/tenant1/exports")
        assert paths.tenant_logs == Path("/app/data/tenant1/logs")
        assert paths.tenant_chromadb == Path("/app/data/tenant1/chromadb")
        assert paths.tenant_shared_documents_lang == Path("/app/data/tenant1/shared/documents/hr")
        assert paths.tenant_shared_processed_lang == Path("/app/data/tenant1/shared/processed/hr")
        assert paths.user_root == Path("/app/data/tenant1/users/user1")
        assert paths.user_documents_lang == Path("/app/data/tenant1/users/user1/documents/hr")
        assert paths.user_processed_lang == Path("/app/data/tenant1/users/user1/processed/hr")
        assert paths.tenant_models_lang == Path("/app/models/tenant1/hr")
        assert paths.tenant_models_embeddings == Path("/app/models/tenant1/hr/embeddings")
        assert paths.tenant_models_generation == Path("/app/models/tenant1/hr/generation")
        assert paths.tenant_models == Path("/app/models/tenant1")
        assert paths.tenant_models_shared == Path("/app/models/tenant1/shared")

    def test_calculate_folder_structure_no_user(self):
        """Test calculating folder structure without user."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="{data_base_dir}/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}_documents"
        )

        paths = calculate_folder_structure(tenant, None, "hr", config)

        assert paths.tenant_root == Path("/app/data/tenant1")
        assert paths.tenant_shared_documents_lang == Path("/app/data/tenant1/shared/documents/hr")
        assert paths.user_root is None
        assert paths.user_documents_lang is None

    def test_calculate_folder_structure_no_language(self):
        """Test calculating folder structure without language."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        user = User(id="1", tenant_id="1", username="user1", email="user1@test.com")
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="{data_base_dir}/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}_documents"
        )

        paths = calculate_folder_structure(tenant, user, None, config)

        assert paths.tenant_root == Path("/app/data/tenant1")
        assert paths.user_root == Path("/app/data/tenant1/users/user1")
        assert paths.tenant_shared_documents_lang is None
        assert paths.user_documents_lang is None
        assert paths.tenant_models_lang is None

    def test_calculate_document_path_user_scope(self):
        """Test calculating document path for user scope."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        user = User(id="1", tenant_id="1", username="user1", email="user1@test.com")
        context = TenantUserContext(tenant=tenant, user=user)
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="{data_base_dir}/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}_documents"
        )

        path = calculate_document_path(context, "hr", DocumentScope.USER, config)

        assert path == Path("/app/data/tenant1/users/user1/documents/hr")

    def test_calculate_document_path_tenant_scope(self):
        """Test calculating document path for tenant scope."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        user = User(id="1", tenant_id="1", username="user1", email="user1@test.com")
        context = TenantUserContext(tenant=tenant, user=user)
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="{data_base_dir}/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}_documents"
        )

        path = calculate_document_path(context, "hr", DocumentScope.TENANT, config)

        assert path == Path("/app/data/tenant1/shared/documents/hr")

    def test_calculate_processed_path_user_scope(self):
        """Test calculating processed path for user scope."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        user = User(id="1", tenant_id="1", username="user1", email="user1@test.com")
        context = TenantUserContext(tenant=tenant, user=user)
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="{data_base_dir}/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}_documents"
        )

        path = calculate_processed_path(context, "hr", DocumentScope.USER, config)

        assert path == Path("/app/data/tenant1/users/user1/processed/hr")

    def test_calculate_processed_path_tenant_scope(self):
        """Test calculating processed path for tenant scope."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        user = User(id="1", tenant_id="1", username="user1", email="user1@test.com")
        context = TenantUserContext(tenant=tenant, user=user)
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="{data_base_dir}/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}_documents"
        )

        path = calculate_processed_path(context, "hr", DocumentScope.TENANT, config)

        assert path == Path("/app/data/tenant1/shared/processed/hr")

    def test_calculate_chromadb_path(self):
        """Test calculating ChromaDB path."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="{data_base_dir}/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}_documents"
        )

        path = calculate_chromadb_path(tenant, config)

        assert path == Path("/app/data/tenant1/chromadb")

    def test_calculate_models_path(self):
        """Test calculating models path."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="{data_base_dir}/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}_documents"
        )

        path = calculate_models_path(tenant, "hr", "embeddings", config)

        assert path == Path("/app/models/tenant1/hr/embeddings")

    def test_calculate_collection_paths(self):
        """Test calculating collection paths."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        user = User(id="1", tenant_id="1", username="user1", email="user1@test.com")
        context = TenantUserContext(tenant=tenant, user=user)
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="{data_base_dir}/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}_documents"
        )

        paths = calculate_collection_paths(context, "hr", config)

        assert paths.user_collection_name == "tenant1_user_hr_documents"
        assert paths.tenant_collection_name == "tenant1_tenant_hr_documents"
        assert paths.base_path == Path("/app/data/tenant1/chromadb")
        assert paths.user_collection_path == Path("/app/data/tenant1/chromadb/tenant1_user_hr_documents")
        assert paths.tenant_collection_path == Path("/app/data/tenant1/chromadb/tenant1_tenant_hr_documents")

    def test_get_system_paths(self):
        """Test getting system paths."""
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="",
            user_documents_template="",
            tenant_shared_template="",
            user_processed_template="",
            tenant_processed_template="",
            chromadb_path_template="",
            models_path_template="",
            collection_name_template=""
        )

        paths = get_system_paths(config)

        assert len(paths) == 3
        assert Path("/app/system/logs") in paths
        assert Path("/app/system/backups") in paths
        assert Path("/app/system/temp") in paths

    def test_get_all_folder_paths(self):
        """Test extracting all folder paths from structure."""
        paths = FolderPaths(
            tenant_root=Path("/app/data/tenant1"),
            tenant_cache=Path("/app/data/tenant1/cache"),
            tenant_exports=None,  # This should be excluded
            user_root=Path("/app/data/tenant1/users/user1")
        )

        all_paths = get_all_folder_paths(paths)

        assert len(all_paths) == 3  # Only non-None paths
        assert Path("/app/data/tenant1") in all_paths
        assert Path("/app/data/tenant1/cache") in all_paths
        assert Path("/app/data/tenant1/users/user1") in all_paths

    def test_calculate_tenant_cleanup_paths(self):
        """Test calculating tenant cleanup paths."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="",
            tenant_shared_template="",
            user_processed_template="",
            tenant_processed_template="",
            chromadb_path_template="",
            models_path_template="",
            collection_name_template=""
        )

        cleanup_paths = calculate_tenant_cleanup_paths(tenant, config)

        assert len(cleanup_paths) == 2
        assert Path("/app/data/tenant1") in cleanup_paths
        assert Path("/app/models/tenant1") in cleanup_paths

    def test_calculate_usage_stats_paths(self):
        """Test calculating usage statistics paths."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        config = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="",
            tenant_shared_template="",
            user_processed_template="",
            tenant_processed_template="",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="",
            collection_name_template=""
        )

        stats_paths = calculate_usage_stats_paths(tenant, config)

        assert len(stats_paths) == 4
        assert stats_paths["documents"] == Path("/app/data/tenant1/users")
        assert stats_paths["processed"] == Path("/app/data/tenant1/shared")
        assert stats_paths["chromadb"] == Path("/app/data/tenant1/chromadb")
        assert stats_paths["models"] == Path("/app/models/tenant1")


class TestTenantFolderManager:
    """Test _TenantFolderManager class."""

    def create_test_providers(self):
        """Create test providers for testing."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_folder_config.return_value = FolderConfig(
            data_base_dir="/app/data",
            models_base_dir="/app/models",
            system_dir="/app/system",
            tenant_root_template="{data_base_dir}/{tenant_slug}",
            user_documents_template="{data_base_dir}/{tenant_slug}/users/{user_id}/documents/{language}",
            tenant_shared_template="{data_base_dir}/{tenant_slug}/shared/documents/{language}",
            user_processed_template="{data_base_dir}/{tenant_slug}/users/{user_id}/processed/{language}",
            tenant_processed_template="{data_base_dir}/{tenant_slug}/shared/processed/{language}",
            chromadb_path_template="{data_base_dir}/{tenant_slug}/chromadb",
            models_path_template="{models_base_dir}/{tenant_slug}/{language}",
            collection_name_template="{tenant_slug}_{scope}_{language}_documents"
        )

        filesystem_provider = Mock(spec=FileSystemProvider)
        filesystem_provider.create_folder.return_value = True
        filesystem_provider.folder_exists.return_value = True
        filesystem_provider.remove_folder.return_value = True
        filesystem_provider.get_folder_stats.return_value = FolderStats(count=5, size_bytes=1024)

        logger_provider = Mock(spec=LoggerProvider)

        return config_provider, filesystem_provider, logger_provider

    def create_test_tenant_user(self):
        """Create test tenant and user."""
        tenant = Tenant(id="1", name="Test Tenant", slug="tenant1", description="Test")
        tenant.get_supported_languages = Mock(return_value=["hr", "en"])
        user = User(id="1", tenant_id="1", username="user1", email="user1@test.com")
        return tenant, user

    def test_tenant_folder_manager_initialization(self):
        """Test tenant folder manager initialization."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()

        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)

        assert manager._config_provider == config_provider
        assert manager._filesystem_provider == filesystem_provider
        assert manager._logger == logger_provider
        assert manager._config is None  # Lazy loading

    def test_tenant_folder_manager_without_logger(self):
        """Test tenant folder manager initialization without logger."""
        config_provider, filesystem_provider, _ = self.create_test_providers()

        manager = _TenantFolderManager(config_provider, filesystem_provider, None)

        assert manager._logger is None

    def test_get_config_caching(self):
        """Test configuration caching."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)

        # First call should load config
        config1 = manager._get_config()
        # Second call should use cached config
        config2 = manager._get_config()

        assert config1 == config2
        config_provider.get_folder_config.assert_called_once()

    def test_get_tenant_folder_structure(self):
        """Test getting tenant folder structure."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, user = self.create_test_tenant_user()

        paths = manager.get_tenant_folder_structure(tenant, user, "hr")

        assert isinstance(paths, FolderPaths)
        assert paths.tenant_root == Path("/app/data/tenant1")
        assert paths.user_documents_lang == Path("/app/data/tenant1/users/user1/documents/hr")

    def test_create_tenant_folder_structure_success(self):
        """Test creating tenant folder structure successfully."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, user = self.create_test_tenant_user()

        success, created_folders = manager.create_tenant_folder_structure(tenant, user, ["hr"])

        assert success is True
        assert len(created_folders) > 0
        assert filesystem_provider.create_folder.call_count > 0

    def test_create_tenant_folder_structure_with_default_languages(self):
        """Test creating folder structure with default languages from tenant."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, user = self.create_test_tenant_user()

        success, created_folders = manager.create_tenant_folder_structure(tenant, user, None)

        assert success is True
        tenant.get_supported_languages.assert_called_once()

    def test_create_tenant_folder_structure_exception(self):
        """Test creating folder structure with exception."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        filesystem_provider.create_folder.side_effect = Exception("Filesystem error")
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, user = self.create_test_tenant_user()

        with pytest.raises(Exception, match="Filesystem error"):
            manager.create_tenant_folder_structure(tenant, user, ["hr"])

    def test_get_tenant_document_path_create_missing(self):
        """Test getting document path with create_if_missing=True."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, user = self.create_test_tenant_user()
        context = TenantUserContext(tenant=tenant, user=user)

        path = manager.get_tenant_document_path(context, "hr", DocumentScope.USER, create_if_missing=True)

        assert path == Path("/app/data/tenant1/users/user1/documents/hr")
        filesystem_provider.create_folder.assert_called_with(path)

    def test_get_tenant_document_path_no_create(self):
        """Test getting document path with create_if_missing=False."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, user = self.create_test_tenant_user()
        context = TenantUserContext(tenant=tenant, user=user)

        path = manager.get_tenant_document_path(context, "hr", DocumentScope.USER, create_if_missing=False)

        assert path == Path("/app/data/tenant1/users/user1/documents/hr")
        # Should not call create_folder
        filesystem_provider.create_folder.assert_not_called()

    def test_get_tenant_processed_path(self):
        """Test getting processed data path."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, user = self.create_test_tenant_user()
        context = TenantUserContext(tenant=tenant, user=user)

        path = manager.get_tenant_processed_path(context, "hr", DocumentScope.TENANT)

        assert path == Path("/app/data/tenant1/shared/processed/hr")
        filesystem_provider.create_folder.assert_called_with(path)

    def test_get_tenant_chromadb_path(self):
        """Test getting ChromaDB path."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, _ = self.create_test_tenant_user()

        path = manager.get_tenant_chromadb_path(tenant)

        assert path == Path("/app/data/tenant1/chromadb")
        filesystem_provider.create_folder.assert_called_with(path)

    def test_get_tenant_models_path(self):
        """Test getting models path."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, _ = self.create_test_tenant_user()

        path = manager.get_tenant_models_path(tenant, "hr", "embeddings")

        assert path == Path("/app/models/tenant1/hr/embeddings")
        filesystem_provider.create_folder.assert_called_with(path)

    def test_get_collection_storage_paths(self):
        """Test getting collection storage paths."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, user = self.create_test_tenant_user()
        context = TenantUserContext(tenant=tenant, user=user)

        paths = manager.get_collection_storage_paths(context, "hr")

        assert isinstance(paths, CollectionPaths)
        assert paths.user_collection_name == "tenant1_user_hr_documents"
        assert paths.tenant_collection_name == "tenant1_tenant_hr_documents"

    def test_ensure_context_folders_success(self):
        """Test ensuring context folders successfully."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, user = self.create_test_tenant_user()
        context = TenantUserContext(tenant=tenant, user=user)

        result = manager.ensure_context_folders(context, "hr")

        assert result is True

    def test_ensure_context_folders_exception(self):
        """Test ensuring context folders with exception."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        filesystem_provider.create_folder.side_effect = Exception("Filesystem error")
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, user = self.create_test_tenant_user()
        context = TenantUserContext(tenant=tenant, user=user)

        with pytest.raises(Exception, match="Filesystem error"):
            manager.ensure_context_folders(context, "hr")

    def test_cleanup_tenant_folders_success(self):
        """Test cleaning up tenant folders successfully."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, _ = self.create_test_tenant_user()

        result = manager.cleanup_tenant_folders(tenant, confirm=True)

        assert result is True
        filesystem_provider.folder_exists.assert_called()
        filesystem_provider.remove_folder.assert_called()

    def test_cleanup_tenant_folders_no_confirm(self):
        """Test cleaning up tenant folders without confirmation."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, _ = self.create_test_tenant_user()

        result = manager.cleanup_tenant_folders(tenant, confirm=False)

        assert result is False
        filesystem_provider.remove_folder.assert_not_called()

    def test_cleanup_tenant_folders_remove_failure(self):
        """Test cleanup when remove operation fails."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        filesystem_provider.remove_folder.return_value = False
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, _ = self.create_test_tenant_user()

        result = manager.cleanup_tenant_folders(tenant, confirm=True)

        assert result is False

    def test_cleanup_tenant_folders_exception(self):
        """Test cleanup with exception."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        filesystem_provider.folder_exists.side_effect = Exception("Filesystem error")
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, _ = self.create_test_tenant_user()

        with pytest.raises(Exception, match="Filesystem error"):
            manager.cleanup_tenant_folders(tenant, confirm=True)

    def test_get_folder_usage_stats(self):
        """Test getting folder usage statistics."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, _ = self.create_test_tenant_user()

        stats = manager.get_folder_usage_stats(tenant)

        assert isinstance(stats, TenantStats)
        assert stats.documents.count == 5
        assert stats.documents.size_bytes == 1024
        assert filesystem_provider.get_folder_stats.call_count == 4  # Four stat types

    def test_get_folder_usage_stats_exception(self):
        """Test getting usage stats with exception."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        filesystem_provider.folder_exists.side_effect = Exception("Filesystem error")
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)
        tenant, _ = self.create_test_tenant_user()

        with pytest.raises(Exception, match="Filesystem error"):
            manager.get_folder_usage_stats(tenant)

    def test_logging_methods(self):
        """Test logging methods work correctly."""
        config_provider, filesystem_provider, logger_provider = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, logger_provider)

        # These should not raise errors
        manager._log_info("test info")
        manager._log_debug("test debug")
        manager._log_warning("test warning")
        manager._log_error("test error")

        # Verify logger was called
        logger_provider.info.assert_called_with("test info")
        logger_provider.debug.assert_called_with("test debug")
        logger_provider.warning.assert_called_with("test warning")
        logger_provider.error.assert_called_with("test error")

    def test_logging_methods_without_logger(self):
        """Test logging methods work without logger."""
        config_provider, filesystem_provider, _ = self.create_test_providers()
        manager = _TenantFolderManager(config_provider, filesystem_provider, None)

        # These should not raise errors even without logger
        manager._log_info("test info")
        manager._log_debug("test debug")
        manager._log_warning("test warning")
        manager._log_error("test error")


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_tenant_folder_manager(self):
        """Test tenant folder manager factory function."""
        config_provider = Mock(spec=ConfigProvider)
        filesystem_provider = Mock(spec=FileSystemProvider)
        logger_provider = Mock(spec=LoggerProvider)

        manager = create_tenant_folder_manager(config_provider, filesystem_provider, logger_provider)

        assert isinstance(manager, _TenantFolderManager)
        assert manager._config_provider == config_provider
        assert manager._filesystem_provider == filesystem_provider
        assert manager._logger == logger_provider

    def test_tenant_folder_manager_public_interface_with_providers(self):
        """Test public TenantFolderManager interface with explicit providers."""
        config_provider = Mock(spec=ConfigProvider)
        filesystem_provider = Mock(spec=FileSystemProvider)
        logger_provider = Mock(spec=LoggerProvider)

        manager = TenantFolderManager(
            config_provider=config_provider,
            filesystem_provider=filesystem_provider,
            logger_provider=logger_provider
        )

        assert isinstance(manager, _TenantFolderManager)
        assert manager._config_provider == config_provider
        assert manager._filesystem_provider == filesystem_provider


if __name__ == "__main__":
    pytest.main([__file__])