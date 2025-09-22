"""
Comprehensive tests for database provider implementations.
Tests both SurrealDBProvider and SupabaseProvider against the DatabaseProvider protocol.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.database.protocols import DatabaseProvider
from src.database.providers.surrealdb_provider import SurrealDBProvider
from src.database.providers.supabase_provider import SupabaseProvider
from src.database.factory import create_database_provider
from src.utils.error_handler import StorageError as DatabaseError, ConfigurationError
from src.models.multitenant_models import (
    Tenant, User, Document, Chunk,
    TenantStatus, BusinessContext, SubscriptionTier,
    UserRole, UserStatus, Language, DocumentStatus, DocumentScope
)


class TestDatabaseProviderProtocol:
    """Test that providers correctly implement the DatabaseProvider protocol."""

    def test_surrealdb_provider_implements_protocol(self):
        """Test SurrealDBProvider implements DatabaseProvider protocol."""
        provider = SurrealDBProvider()
        assert isinstance(provider, DatabaseProvider)

    def test_supabase_provider_implements_protocol(self):
        """Test SupabaseProvider implements DatabaseProvider protocol."""
        provider = SupabaseProvider()
        assert isinstance(provider, DatabaseProvider)


class TestDatabaseFactory:
    """Test database provider factory functionality."""

    def test_create_surrealdb_provider(self):
        """Test factory creates SurrealDBProvider correctly."""
        config = {"provider": "surrealdb"}
        provider = create_database_provider(config)
        assert isinstance(provider, SurrealDBProvider)

    def test_create_supabase_provider(self):
        """Test factory creates SupabaseProvider correctly."""
        config = {"provider": "supabase"}
        provider = create_database_provider(config)
        assert isinstance(provider, SupabaseProvider)

    def test_factory_invalid_provider(self):
        """Test factory raises error for invalid provider."""
        config = {"provider": "invalid"}
        with pytest.raises(ConfigurationError, match="Unsupported database provider"):
            create_database_provider(config)

    def test_factory_missing_provider(self):
        """Test factory raises error when provider not specified."""
        config = {}
        with pytest.raises(ConfigurationError, match="Database provider must be specified"):
            create_database_provider(config)


class TestSurrealDBProvider:
    """Test SurrealDBProvider implementation."""

    @pytest.fixture
    def provider(self):
        """Create SurrealDBProvider instance."""
        return SurrealDBProvider()

    @pytest.fixture
    def mock_surrealdb_config(self):
        """Mock SurrealDB configuration."""
        return {
            "surrealdb": {
                "url": "ws://localhost:8000",
                "namespace": "test",
                "database": "rag",
                "username": "root",
                "password": "root"
            }
        }

    @pytest.fixture
    def sample_tenant(self):
        """Create sample tenant for testing."""
        return Tenant(
            name="Test Tenant",
            slug="test-tenant",
            description="Test tenant description",
            status=TenantStatus.ACTIVE,
            language_preference=Language.HR,
            cultural_context=BusinessContext.CROATIAN_BUSINESS,
            subscription_tier=SubscriptionTier.BASIC,
            settings={"key": "value"}
        )

    @pytest.fixture
    def sample_user(self):
        """Create sample user for testing."""
        return User(
            tenant_id="tenant_123",
            email="test@example.com",
            username="testuser",
            full_name="Test User",
            password_hash="hashed_password",
            role=UserRole.MEMBER,
            status=UserStatus.ACTIVE,
            language_preference=Language.HR,
            settings={"theme": "dark"}
        )

    async def test_initialize_success(self, provider, mock_surrealdb_config):
        """Test successful SurrealDB initialization."""
        with patch('src.database.providers.surrealdb_provider.Surreal') as mock_surreal:
            mock_client = AsyncMock()
            mock_surreal.return_value = mock_client
            mock_client.query.return_value = [{"test": 1}]

            await provider.initialize(mock_surrealdb_config)

            assert provider.client is not None
            mock_client.signin.assert_called_once()
            mock_client.use.assert_called_once()
            mock_client.query.assert_called_once()

    async def test_initialize_missing_config(self, provider):
        """Test initialization fails with missing config section."""
        config = {}
        with pytest.raises(DatabaseError, match="Missing SurrealDB configuration section"):
            await provider.initialize(config)

    async def test_initialize_missing_required_keys(self, provider):
        """Test initialization fails with missing required keys."""
        config = {"surrealdb": {"url": "ws://localhost:8000"}}
        with pytest.raises(DatabaseError, match="Missing required SurrealDB config keys"):
            await provider.initialize(config)

    async def test_initialize_import_error(self, provider, mock_surrealdb_config):
        """Test initialization fails when SurrealDB not installed."""
        with patch('src.database.providers.surrealdb_provider.Surreal', side_effect=ImportError()):
            with pytest.raises(DatabaseError, match="SurrealDB not installed"):
                await provider.initialize(mock_surrealdb_config)

    async def test_health_check_success(self, provider):
        """Test successful health check."""
        provider.client = AsyncMock()
        provider.client.query.return_value = [{"health_check": 1}]

        result = await provider.health_check()
        assert result is True

    async def test_health_check_no_client(self, provider):
        """Test health check fails when client not initialized."""
        result = await provider.health_check()
        assert result is False

    async def test_health_check_query_failure(self, provider):
        """Test health check fails when query fails."""
        provider.client = AsyncMock()
        provider.client.query.side_effect = Exception("Connection failed")

        result = await provider.health_check()
        assert result is False

    async def test_create_tenant_success(self, provider, sample_tenant):
        """Test successful tenant creation."""
        provider.client = AsyncMock()
        mock_result = [{
            "id": "tenant_123",
            "name": "Test Tenant",
            "slug": "test-tenant",
            "status": "active",
            "language_preference": "hr",
            "cultural_context": "croatian_business",
            "subscription_tier": "basic"
        }]
        provider.client.query.return_value = mock_result

        result = await provider.create_tenant(sample_tenant)

        assert result.id == "tenant_123"
        assert result.name == "Test Tenant"
        assert result.slug == "test-tenant"

    async def test_create_tenant_no_results(self, provider, sample_tenant):
        """Test tenant creation fails when no results returned."""
        provider.client = AsyncMock()
        provider.client.query.return_value = []

        with pytest.raises(DatabaseError, match="Tenant creation returned no results"):
            await provider.create_tenant(sample_tenant)

    async def test_get_tenant_success(self, provider):
        """Test successful tenant retrieval."""
        provider.client = AsyncMock()
        mock_result = [{
            "id": "tenant_123",
            "name": "Test Tenant",
            "slug": "test-tenant",
            "status": "active",
            "language_preference": "hr",
            "cultural_context": "croatian_business",
            "subscription_tier": "basic"
        }]
        provider.client.query.return_value = mock_result

        result = await provider.get_tenant("tenant_123")

        assert result.id == "tenant_123"
        assert result.name == "Test Tenant"

    async def test_get_tenant_not_found(self, provider):
        """Test tenant retrieval fails when tenant not found."""
        provider.client = AsyncMock()
        provider.client.query.return_value = []

        with pytest.raises(DatabaseError, match="Tenant not found"):
            await provider.get_tenant("nonexistent")

    async def test_create_user_success(self, provider, sample_user):
        """Test successful user creation."""
        provider.client = AsyncMock()
        mock_result = [{
            "id": "user_123",
            "tenant_id": "tenant_123",
            "email": "test@example.com",
            "username": "testuser",
            "password_hash": "hashed_password",
            "role": "member",
            "status": "active",
            "language_preference": "hr"
        }]
        provider.client.query.return_value = mock_result

        result = await provider.create_user(sample_user)

        assert result.id == "user_123"
        assert result.email == "test@example.com"
        assert result.username == "testuser"

    def test_parse_tenant_result_success(self, provider):
        """Test successful tenant result parsing."""
        result_data = {
            "id": "tenant_123",
            "name": "Test Tenant",
            "slug": "test-tenant",
            "status": "active",
            "language_preference": "hr",
            "cultural_context": "croatian_business",
            "subscription_tier": "basic"
        }

        tenant = provider._parse_tenant_result(result_data)

        assert tenant.id == "tenant_123"
        assert tenant.name == "Test Tenant"
        assert tenant.status == TenantStatus.ACTIVE

    def test_parse_tenant_result_missing_field(self, provider):
        """Test tenant parsing fails with missing required field."""
        result_data = {"name": "Test Tenant"}

        with pytest.raises(DatabaseError, match="Missing required field 'id'"):
            provider._parse_tenant_result(result_data)

    def test_parse_user_result_success(self, provider):
        """Test successful user result parsing."""
        result_data = {
            "id": "user_123",
            "tenant_id": "tenant_123",
            "email": "test@example.com",
            "username": "testuser",
            "password_hash": "hashed_password",
            "role": "member",
            "status": "active",
            "language_preference": "hr"
        }

        user = provider._parse_user_result(result_data)

        assert user.id == "user_123"
        assert user.email == "test@example.com"
        assert user.role == UserRole.MEMBER

    def test_parse_user_result_missing_field(self, provider):
        """Test user parsing fails with missing required field."""
        result_data = {"email": "test@example.com"}

        with pytest.raises(DatabaseError, match="Missing required field 'id'"):
            provider._parse_user_result(result_data)


class TestSupabaseProvider:
    """Test SupabaseProvider implementation."""

    @pytest.fixture
    def provider(self):
        """Create SupabaseProvider instance."""
        return SupabaseProvider()

    @pytest.fixture
    def mock_supabase_config(self):
        """Mock Supabase configuration."""
        return {
            "supabase": {
                "url": "https://test.supabase.co",
                "service_role_key": "service_key",
                "anon_key": "anon_key",
                "enable_rls": True
            }
        }

    async def test_initialize_success(self, provider, mock_supabase_config):
        """Test successful Supabase initialization."""
        with patch('src.database.providers.supabase_provider.create_client') as mock_create_client:
            mock_client = Mock()
            mock_table = Mock()
            mock_table.select.return_value.limit.return_value.execute.return_value.data = [{"table_name": "test"}]
            mock_client.table.return_value = mock_table
            mock_create_client.return_value = mock_client

            await provider.initialize(mock_supabase_config)

            assert provider.client is not None
            mock_create_client.assert_called_once()

    async def test_initialize_missing_config(self, provider):
        """Test initialization fails with missing config section."""
        config = {}
        with pytest.raises(DatabaseError, match="Missing Supabase configuration section"):
            await provider.initialize(config)

    async def test_initialize_missing_required_keys(self, provider):
        """Test initialization fails with missing required keys."""
        config = {"supabase": {"url": "https://test.supabase.co"}}
        with pytest.raises(DatabaseError, match="Missing required Supabase config keys"):
            await provider.initialize(config)

    async def test_initialize_import_error(self, provider, mock_supabase_config):
        """Test initialization fails when Supabase not installed."""
        with patch('src.database.providers.supabase_provider.create_client', side_effect=ImportError()):
            with pytest.raises(DatabaseError, match="Supabase not installed"):
                await provider.initialize(mock_supabase_config)

    async def test_health_check_success(self, provider):
        """Test successful health check."""
        mock_client = Mock()
        mock_table = Mock()
        mock_table.select.return_value.limit.return_value.execute.return_value.data = [{"count": 1}]
        mock_client.table.return_value = mock_table
        provider.client = mock_client

        result = await provider.health_check()
        assert result is True

    async def test_health_check_no_client(self, provider):
        """Test health check fails when client not initialized."""
        result = await provider.health_check()
        assert result is False

    async def test_create_tenant_success(self, provider):
        """Test successful tenant creation with RLS."""
        mock_client = Mock()
        mock_table = Mock()
        mock_result = Mock()
        mock_result.data = [{
            "id": "tenant_123",
            "name": "Test Tenant",
            "slug": "test-tenant",
            "status": "active",
            "language_preference": "hr",
            "cultural_context": "croatian_business",
            "subscription_tier": "basic"
        }]
        mock_table.insert.return_value.execute.return_value = mock_result
        mock_client.table.return_value = mock_table
        provider.client = mock_client

        sample_tenant = Tenant(
            name="Test Tenant",
            slug="test-tenant",
            status=TenantStatus.ACTIVE,
            language_preference=Language.HR,
            cultural_context=BusinessContext.CROATIAN_BUSINESS,
            subscription_tier=SubscriptionTier.BASIC
        )

        result = await provider.create_tenant(sample_tenant)

        assert result.id == "tenant_123"
        assert result.name == "Test Tenant"

    def test_parse_tenant_result_success(self, provider):
        """Test successful tenant result parsing."""
        result_data = {
            "id": "tenant_123",
            "name": "Test Tenant",
            "slug": "test-tenant",
            "status": "active",
            "language_preference": "hr",
            "cultural_context": "croatian_business",
            "subscription_tier": "basic"
        }

        tenant = provider._parse_tenant_result(result_data)

        assert tenant.id == "tenant_123"
        assert tenant.name == "Test Tenant"
        assert tenant.status == TenantStatus.ACTIVE

    def test_parse_tenant_result_missing_field(self, provider):
        """Test tenant parsing fails with missing required field."""
        result_data = {"name": "Test Tenant"}

        with pytest.raises(DatabaseError, match="Missing required field 'id'"):
            provider._parse_tenant_result(result_data)


class TestDocumentOperations:
    """Test document operations across both providers."""

    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing."""
        return Document(
            tenant_id="tenant_123",
            user_id="user_123",
            title="Test Document",
            content="This is test content",
            language=Language.HR,
            scope=DocumentScope.USER,
            category="test",
            status=DocumentStatus.PENDING,
            metadata={"source": "test"}
        )

    @pytest.fixture
    def sample_chunk(self):
        """Create sample chunk for testing."""
        return Chunk(
            document_id="doc_123",
            tenant_id="tenant_123",
            user_id="user_123",
            chunk_index=0,
            content="Chunk content",
            language=Language.HR,
            start_char=0,
            end_char=100,
            token_count=20,
            vector_id="vec_123",
            vector_collection="test_collection"
        )

    async def test_surrealdb_create_document(self, sample_document):
        """Test document creation in SurrealDB."""
        provider = SurrealDBProvider()
        provider.client = AsyncMock()
        mock_result = [{
            "id": "doc_123",
            "tenant_id": "tenant_123",
            "user_id": "user_123",
            "title": "Test Document",
            "content": "This is test content",
            "language": "hr",
            "scope": "user",
            "status": "pending"
        }]
        provider.client.query.return_value = mock_result

        result = await provider.create_document(sample_document)

        assert result.id == "doc_123"
        assert result.title == "Test Document"

    async def test_supabase_create_document(self, sample_document):
        """Test document creation in Supabase."""
        provider = SupabaseProvider()
        mock_client = Mock()
        mock_table = Mock()
        mock_result = Mock()
        mock_result.data = [{
            "id": "doc_123",
            "tenant_id": "tenant_123",
            "user_id": "user_123",
            "title": "Test Document",
            "content": "This is test content",
            "language": "hr",
            "scope": "user",
            "status": "pending"
        }]
        mock_table.insert.return_value.execute.return_value = mock_result
        mock_client.table.return_value = mock_table
        provider.client = mock_client

        result = await provider.create_document(sample_document)

        assert result.id == "doc_123"
        assert result.title == "Test Document"

    async def test_surrealdb_create_chunk(self, sample_chunk):
        """Test chunk creation in SurrealDB."""
        provider = SurrealDBProvider()
        provider.client = AsyncMock()
        mock_result = [{
            "id": "chunk_123",
            "document_id": "doc_123",
            "tenant_id": "tenant_123",
            "user_id": "user_123",
            "chunk_index": 0,
            "content": "Chunk content",
            "language": "hr"
        }]
        provider.client.query.return_value = mock_result

        result = await provider.create_chunk(sample_chunk)

        assert result.id == "chunk_123"
        assert result.document_id == "doc_123"
        assert result.chunk_index == 0

    async def test_supabase_create_chunk(self, sample_chunk):
        """Test chunk creation in Supabase."""
        provider = SupabaseProvider()
        mock_client = Mock()
        mock_table = Mock()
        mock_result = Mock()
        mock_result.data = [{
            "id": "chunk_123",
            "document_id": "doc_123",
            "tenant_id": "tenant_123",
            "user_id": "user_123",
            "chunk_index": 0,
            "content": "Chunk content",
            "language": "hr"
        }]
        mock_table.insert.return_value.execute.return_value = mock_result
        mock_client.table.return_value = mock_table
        provider.client = mock_client

        result = await provider.create_chunk(sample_chunk)

        assert result.id == "chunk_123"
        assert result.document_id == "doc_123"
        assert result.chunk_index == 0


class TestConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_surrealdb_config_validation(self):
        """Test SurrealDB configuration validation in factory."""
        from src.database.factory import validate_provider_config

        config = {
            "surrealdb": {
                "url": "ws://localhost:8000",
                "namespace": "test",
                "database": "rag",
                "username": "root",
                "password": "root"
            }
        }

        validate_provider_config(config, "surrealdb")

    def test_supabase_config_validation(self):
        """Test Supabase configuration validation in factory."""
        from src.database.factory import validate_provider_config

        config = {
            "supabase": {
                "url": "https://test.supabase.co",
                "service_role_key": "service_key",
                "anon_key": "anon_key"
            }
        }

        validate_provider_config(config, "supabase")

    def test_surrealdb_missing_config_validation(self):
        """Test SurrealDB configuration validation fails with missing keys."""
        from src.database.factory import validate_provider_config

        config = {"surrealdb": {"url": "ws://localhost:8000"}}

        with pytest.raises(ConfigurationError, match="Missing required surrealdb config keys"):
            validate_provider_config(config, "surrealdb")

    def test_supabase_missing_config_validation(self):
        """Test Supabase configuration validation fails with missing keys."""
        from src.database.factory import validate_provider_config

        config = {"supabase": {"url": "https://test.supabase.co"}}

        with pytest.raises(ConfigurationError, match="Missing required supabase config keys"):
            validate_provider_config(config, "supabase")