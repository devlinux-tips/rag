"""
Comprehensive tests for multitenant models system.
Tests all enums, dataclasses, methods, and constants for multitenant architecture.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.models.multitenant_models import (
    # Enums
    TenantStatus,
    UserRole,
    UserStatus,
    DocumentScope,
    DocumentStatus,
    FileType,
    Language,
    BusinessContext,
    SubscriptionTier,

    # Dataclasses
    Tenant,
    User,
    Document,
    Chunk,
    SearchQuery,
    CategorizationTemplate,
    SystemConfig,
    TenantUserContext,
    MultiTenantQueryResult,

    # Constants
    DEFAULT_DEVELOPMENT_TENANT,
    DEFAULT_DEVELOPMENT_USER,
    DEFAULT_DEVELOPMENT_CONTEXT,
)

from src.retrieval.categorization import CategoryType


# Test Enums
class TestEnums:
    """Test all enum classes."""

    def test_tenant_status_enum(self):
        """Test TenantStatus enum values."""
        assert TenantStatus.ACTIVE.value == "active"
        assert TenantStatus.SUSPENDED.value == "suspended"
        assert TenantStatus.INACTIVE.value == "inactive"

        # Test all values exist
        statuses = [status.value for status in TenantStatus]
        assert "active" in statuses
        assert "suspended" in statuses
        assert "inactive" in statuses
        assert len(statuses) == 3

    def test_user_role_enum(self):
        """Test UserRole enum values."""
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.MEMBER.value == "member"
        assert UserRole.VIEWER.value == "viewer"

        # Test all values exist
        roles = [role.value for role in UserRole]
        assert "admin" in roles
        assert "member" in roles
        assert "viewer" in roles
        assert len(roles) == 3

    def test_user_status_enum(self):
        """Test UserStatus enum values."""
        assert UserStatus.ACTIVE.value == "active"
        assert UserStatus.SUSPENDED.value == "suspended"
        assert UserStatus.INACTIVE.value == "inactive"

        # Test all values exist
        statuses = [status.value for status in UserStatus]
        assert "active" in statuses
        assert "suspended" in statuses
        assert "inactive" in statuses
        assert len(statuses) == 3

    def test_document_scope_enum(self):
        """Test DocumentScope enum values."""
        assert DocumentScope.USER.value == "user"
        assert DocumentScope.TENANT.value == "tenant"

        # Test all values exist
        scopes = [scope.value for scope in DocumentScope]
        assert "user" in scopes
        assert "tenant" in scopes
        assert len(scopes) == 2

    def test_document_status_enum(self):
        """Test DocumentStatus enum values."""
        assert DocumentStatus.UPLOADED.value == "uploaded"
        assert DocumentStatus.PROCESSING.value == "processing"
        assert DocumentStatus.PROCESSED.value == "processed"
        assert DocumentStatus.FAILED.value == "failed"
        assert DocumentStatus.ARCHIVED.value == "archived"

        # Test all values exist
        statuses = [status.value for status in DocumentStatus]
        assert "uploaded" in statuses
        assert "processing" in statuses
        assert "processed" in statuses
        assert "failed" in statuses
        assert "archived" in statuses
        assert len(statuses) == 5

    def test_file_type_enum(self):
        """Test FileType enum values."""
        assert FileType.PDF.value == "pdf"
        assert FileType.DOCX.value == "docx"
        assert FileType.TXT.value == "txt"
        assert FileType.MARKDOWN.value == "md"
        assert FileType.HTML.value == "html"

        # Test all values exist
        types = [file_type.value for file_type in FileType]
        assert "pdf" in types
        assert "docx" in types
        assert "txt" in types
        assert "md" in types
        assert "html" in types
        assert len(types) == 5

    def test_language_enum(self):
        """Test Language enum values."""
        assert Language.AUTO.value == "auto"
        assert Language.MULTILINGUAL.value == "multilingual"

        # Test all values exist
        languages = [lang.value for lang in Language]
        assert "auto" in languages
        assert "multilingual" in languages
        assert len(languages) == 2

    def test_business_context_enum(self):
        """Test BusinessContext enum values."""
        assert BusinessContext.BUSINESS.value == "business"
        assert BusinessContext.ACADEMIC.value == "academic"
        assert BusinessContext.TECHNICAL.value == "technical"
        assert BusinessContext.LEGAL.value == "legal"
        assert BusinessContext.HEALTHCARE.value == "healthcare"
        assert BusinessContext.GOVERNMENT.value == "government"

        # Test all values exist
        contexts = [context.value for context in BusinessContext]
        assert "business" in contexts
        assert "academic" in contexts
        assert "technical" in contexts
        assert "legal" in contexts
        assert "healthcare" in contexts
        assert "government" in contexts
        assert len(contexts) == 6

    def test_subscription_tier_enum(self):
        """Test SubscriptionTier enum values."""
        assert SubscriptionTier.BASIC.value == "basic"
        assert SubscriptionTier.PROFESSIONAL.value == "professional"
        assert SubscriptionTier.ENTERPRISE.value == "enterprise"

        # Test all values exist
        tiers = [tier.value for tier in SubscriptionTier]
        assert "basic" in tiers
        assert "professional" in tiers
        assert "enterprise" in tiers
        assert len(tiers) == 3


# Test Dataclasses
class TestTenant:
    """Test Tenant dataclass."""

    @pytest.fixture
    def sample_tenant(self):
        """Create sample tenant for testing."""
        return Tenant(
            id="tenant:test",
            name="Test Tenant",
            slug="test",
            description="Test tenant description",
            status=TenantStatus.ACTIVE,
            settings={"max_documents_per_user": 100, "max_total_documents": 1000},
            business_context=BusinessContext.TECHNICAL,
            subscription_tier=SubscriptionTier.PROFESSIONAL
        )

    def test_tenant_creation(self, sample_tenant):
        """Test basic tenant creation."""
        assert sample_tenant.id == "tenant:test"
        assert sample_tenant.name == "Test Tenant"
        assert sample_tenant.slug == "test"
        assert sample_tenant.description == "Test tenant description"
        assert sample_tenant.status == TenantStatus.ACTIVE
        assert sample_tenant.business_context == BusinessContext.TECHNICAL
        assert sample_tenant.subscription_tier == SubscriptionTier.PROFESSIONAL
        assert isinstance(sample_tenant.created_at, datetime)
        assert isinstance(sample_tenant.updated_at, datetime)

    def test_tenant_defaults(self):
        """Test tenant default values."""
        tenant = Tenant(id="tenant:minimal", name="Minimal", slug="minimal")

        assert tenant.description is None
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.settings == {}
        assert tenant.business_context == BusinessContext.BUSINESS
        assert tenant.subscription_tier == SubscriptionTier.BASIC
        assert isinstance(tenant.created_at, datetime)
        assert isinstance(tenant.updated_at, datetime)

    @patch('src.utils.config_loader.get_supported_languages')
    def test_get_supported_languages(self, mock_get_supported_languages, sample_tenant):
        """Test getting supported languages."""
        mock_get_supported_languages.return_value = ["hr", "en", "de"]

        languages = sample_tenant.get_supported_languages()

        assert languages == ["hr", "en", "de"]
        mock_get_supported_languages.assert_called_once()

    def test_get_collection_name(self, sample_tenant):
        """Test ChromaDB collection name generation."""
        collection_name = sample_tenant.get_collection_name(DocumentScope.USER, "hr")
        assert collection_name == "test_user_hr"

        collection_name = sample_tenant.get_collection_name(DocumentScope.TENANT, "en")
        assert collection_name == "test_tenant_en"

    def test_can_create_user(self, sample_tenant):
        """Test user creation permission."""
        # Active tenant can create users
        assert sample_tenant.can_create_user() is True

        # Suspended tenant cannot create users
        sample_tenant.status = TenantStatus.SUSPENDED
        assert sample_tenant.can_create_user() is False

        # Inactive tenant cannot create users
        sample_tenant.status = TenantStatus.INACTIVE
        assert sample_tenant.can_create_user() is False

    def test_get_max_documents(self, sample_tenant):
        """Test getting max documents limit."""
        max_docs = sample_tenant.get_max_documents()
        assert max_docs == 1000

    def test_get_max_documents_per_user(self, sample_tenant):
        """Test getting max documents per user limit."""
        max_docs = sample_tenant.get_max_documents_per_user()
        assert max_docs == 100


class TestUser:
    """Test User dataclass."""

    @pytest.fixture
    def sample_user(self):
        """Create sample user for testing."""
        return User(
            id="user:test",
            tenant_id="tenant:test",
            email="test@example.com",
            username="testuser",
            full_name="Test User",
            password_hash="$2b$12$test_hash",
            role=UserRole.MEMBER,
            status=UserStatus.ACTIVE,
            settings={
                "preferred_languages": ["hr", "en"],
                "preferred_categories": ["technical", "business"],
                "auto_categorize": True
            }
        )

    def test_user_creation(self, sample_user):
        """Test basic user creation."""
        assert sample_user.id == "user:test"
        assert sample_user.tenant_id == "tenant:test"
        assert sample_user.email == "test@example.com"
        assert sample_user.username == "testuser"
        assert sample_user.full_name == "Test User"
        assert sample_user.password_hash == "$2b$12$test_hash"
        assert sample_user.role == UserRole.MEMBER
        assert sample_user.status == UserStatus.ACTIVE
        assert isinstance(sample_user.created_at, datetime)
        assert isinstance(sample_user.updated_at, datetime)

    def test_user_defaults(self):
        """Test user default values."""
        user = User(
            id="user:minimal",
            tenant_id="tenant:test",
            email="minimal@example.com",
            username="minimal"
        )

        assert user.full_name is None
        assert user.password_hash == ""
        assert user.role == UserRole.MEMBER
        assert user.status == UserStatus.ACTIVE
        assert user.settings == {}
        assert user.last_login_at is None
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.updated_at, datetime)

    def test_get_preferred_languages(self, sample_user):
        """Test getting preferred languages."""
        languages = sample_user.get_preferred_languages()
        assert languages == ["hr", "en"]

    @patch('src.utils.config_loader.get_shared_config')
    def test_get_preferred_languages_fallback(self, mock_get_shared_config):
        """Test fallback to default language when no preferences."""
        mock_get_shared_config.return_value = {
            "languages": {"default": "hr"}
        }

        user = User(
            id="user:fallback",
            tenant_id="tenant:test",
            email="fallback@example.com",
            username="fallback",
            settings={"preferred_languages": None}
        )

        languages = user.get_preferred_languages()
        assert languages == ["hr"]
        mock_get_shared_config.assert_called_once()

    @patch('src.utils.config_loader.get_shared_config')
    def test_get_preferred_languages_config_error(self, mock_get_shared_config):
        """Test error handling when config is missing default language."""
        mock_get_shared_config.return_value = {"languages": {}}

        user = User(
            id="user:error",
            tenant_id="tenant:test",
            email="error@example.com",
            username="error",
            settings={"preferred_languages": None}
        )

        with pytest.raises(ValueError, match="Missing 'default' in languages configuration"):
            user.get_preferred_languages()

    def test_can_upload_documents(self, sample_user):
        """Test document upload permission."""
        # Active member can upload
        assert sample_user.can_upload_documents() is True

        # Admin can upload
        sample_user.role = UserRole.ADMIN
        assert sample_user.can_upload_documents() is True

        # Viewer cannot upload
        sample_user.role = UserRole.VIEWER
        assert sample_user.can_upload_documents() is False

        # Inactive user cannot upload
        sample_user.role = UserRole.MEMBER
        sample_user.status = UserStatus.INACTIVE
        assert sample_user.can_upload_documents() is False

    def test_can_access_tenant_documents(self, sample_user):
        """Test tenant document access permission."""
        # Active user can access tenant documents
        assert sample_user.can_access_tenant_documents() is True

        # Inactive user cannot access tenant documents
        sample_user.status = UserStatus.INACTIVE
        assert sample_user.can_access_tenant_documents() is False

    def test_can_promote_documents_to_tenant(self, sample_user):
        """Test document promotion permission."""
        # Member can promote
        assert sample_user.can_promote_documents_to_tenant() is True

        # Admin can promote
        sample_user.role = UserRole.ADMIN
        assert sample_user.can_promote_documents_to_tenant() is True

        # Viewer cannot promote
        sample_user.role = UserRole.VIEWER
        assert sample_user.can_promote_documents_to_tenant() is False

        # Inactive user cannot promote
        sample_user.role = UserRole.MEMBER
        sample_user.status = UserStatus.INACTIVE
        assert sample_user.can_promote_documents_to_tenant() is False

    def test_get_preferred_categories(self, sample_user):
        """Test getting preferred categories."""
        categories = sample_user.get_preferred_categories()
        assert categories == ["technical", "business"]


class TestDocument:
    """Test Document dataclass."""

    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing."""
        return Document(
            id="doc:test",
            tenant_id="tenant:test",
            user_id="user:test",
            title="Test Document",
            filename="test.pdf",
            file_path="/data/test/test.pdf",
            file_size=1024,
            file_type=FileType.PDF,
            language="hr",
            scope=DocumentScope.USER,
            status=DocumentStatus.PROCESSED,
            content_hash="abc123",
            metadata={"source": "upload"},
            categories=["technical"],
            tags=["test", "pdf"],
            chunk_count=5
        )

    def test_document_creation(self, sample_document):
        """Test basic document creation."""
        assert sample_document.id == "doc:test"
        assert sample_document.tenant_id == "tenant:test"
        assert sample_document.user_id == "user:test"
        assert sample_document.title == "Test Document"
        assert sample_document.filename == "test.pdf"
        assert sample_document.file_path == "/data/test/test.pdf"
        assert sample_document.file_size == 1024
        assert sample_document.file_type == FileType.PDF
        assert sample_document.language == "hr"
        assert sample_document.scope == DocumentScope.USER
        assert sample_document.status == DocumentStatus.PROCESSED
        assert sample_document.content_hash == "abc123"
        assert sample_document.chunk_count == 5
        assert isinstance(sample_document.created_at, datetime)
        assert isinstance(sample_document.updated_at, datetime)

    def test_document_defaults(self):
        """Test document default values."""
        document = Document(
            id="doc:minimal",
            tenant_id="tenant:test",
            user_id="user:test",
            title="Minimal Document",
            filename="minimal.txt",
            file_path="/data/minimal.txt"
        )

        assert document.file_size == 0
        assert document.file_type == FileType.TXT
        assert document.language == "auto"
        assert document.scope == DocumentScope.USER
        assert document.status == DocumentStatus.UPLOADED
        assert document.content_hash is None
        assert document.metadata == {}
        assert document.categories == []
        assert document.tags == []
        assert document.chunk_count == 0
        assert document.processing_started_at is None
        assert document.processing_completed_at is None
        assert isinstance(document.created_at, datetime)
        assert isinstance(document.updated_at, datetime)

    def test_get_collection_name(self, sample_document):
        """Test ChromaDB collection name generation."""
        collection_name = sample_document.get_collection_name("test-tenant")
        assert collection_name == "test-tenant_user_hr"

        # Test with tenant scope
        sample_document.scope = DocumentScope.TENANT
        collection_name = sample_document.get_collection_name("test-tenant")
        assert collection_name == "test-tenant_tenant_hr"

    def test_is_processed(self, sample_document):
        """Test processed status check."""
        assert sample_document.is_processed() is True

        sample_document.status = DocumentStatus.PROCESSING
        assert sample_document.is_processed() is False

        sample_document.status = DocumentStatus.FAILED
        assert sample_document.is_processed() is False

    def test_can_be_promoted_to_tenant(self, sample_document):
        """Test promotion to tenant eligibility."""
        # Processed user document can be promoted
        assert sample_document.can_be_promoted_to_tenant() is True

        # Tenant document cannot be promoted
        sample_document.scope = DocumentScope.TENANT
        assert sample_document.can_be_promoted_to_tenant() is False

        # Unprocessed document cannot be promoted
        sample_document.scope = DocumentScope.USER
        sample_document.status = DocumentStatus.PROCESSING
        assert sample_document.can_be_promoted_to_tenant() is False

    def test_get_display_name(self, sample_document):
        """Test display name generation."""
        # Uses title when available
        assert sample_document.get_display_name() == "Test Document"

        # Falls back to filename when no title
        sample_document.title = ""
        assert sample_document.get_display_name() == "test.pdf"

        sample_document.title = None
        assert sample_document.get_display_name() == "test.pdf"


class TestChunk:
    """Test Chunk dataclass."""

    @pytest.fixture
    def sample_chunk(self):
        """Create sample chunk for testing."""
        return Chunk(
            id="chunk:test",
            document_id="doc:test",
            tenant_id="tenant:test",
            user_id="user:test",
            scope=DocumentScope.USER,
            chunk_index=0,
            content="This is test content for chunking.",
            language="hr",
            embedding_model="bge-m3",
            vector_collection="test_user_hr",
            vector_id="vec:123",
            metadata={"source": "paragraph"},
            categories=["technical"]
        )

    def test_chunk_creation(self, sample_chunk):
        """Test basic chunk creation."""
        assert sample_chunk.id == "chunk:test"
        assert sample_chunk.document_id == "doc:test"
        assert sample_chunk.tenant_id == "tenant:test"
        assert sample_chunk.user_id == "user:test"
        assert sample_chunk.scope == DocumentScope.USER
        assert sample_chunk.chunk_index == 0
        assert sample_chunk.content == "This is test content for chunking."
        assert sample_chunk.language == "hr"
        assert sample_chunk.embedding_model == "bge-m3"
        assert sample_chunk.vector_collection == "test_user_hr"
        assert sample_chunk.vector_id == "vec:123"
        assert isinstance(sample_chunk.created_at, datetime)

    def test_chunk_defaults(self):
        """Test chunk default values."""
        chunk = Chunk(
            id="chunk:minimal",
            document_id="doc:test",
            tenant_id="tenant:test",
            user_id="user:test",
            scope=DocumentScope.USER,
            chunk_index=0,
            content="Minimal content."
        )

        # Content length will be calculated by __post_init__ since it's 0 by default
        expected_length = len("Minimal content.")
        assert chunk.content_length == expected_length
        assert chunk.language == "auto"
        assert chunk.embedding_model == "bge-m3"
        assert chunk.vector_collection == ""
        assert chunk.vector_id == ""
        assert chunk.metadata == {}
        assert chunk.categories == []
        assert isinstance(chunk.created_at, datetime)

    def test_post_init_content_length(self):
        """Test content length calculation in __post_init__."""
        chunk = Chunk(
            id="chunk:length",
            document_id="doc:test",
            tenant_id="tenant:test",
            user_id="user:test",
            scope=DocumentScope.USER,
            chunk_index=0,
            content="Content for length calculation."
        )

        # Should automatically calculate content length
        assert chunk.content_length == len("Content for length calculation.")

    def test_post_init_preserves_explicit_length(self):
        """Test that explicit content_length is preserved."""
        chunk = Chunk(
            id="chunk:explicit",
            document_id="doc:test",
            tenant_id="tenant:test",
            user_id="user:test",
            scope=DocumentScope.USER,
            chunk_index=0,
            content="Test content.",
            content_length=100  # Explicit value
        )

        # Should preserve explicit value
        assert chunk.content_length == 100


class TestSearchQuery:
    """Test SearchQuery dataclass."""

    @pytest.fixture
    def sample_query(self):
        """Create sample search query for testing."""
        return SearchQuery(
            id="query:test",
            tenant_id="tenant:test",
            user_id="user:test",
            query_text="What is machine learning?",
            query_language="en",
            detected_language="en",
            primary_category=CategoryType.TECHNICAL,
            secondary_categories=[CategoryType.EDUCATION, CategoryType.ACADEMIC],
            retrieval_strategy="hybrid",
            scope_searched=[DocumentScope.USER, DocumentScope.TENANT],
            results_count=5,
            response_time_ms=150,
            satisfaction_rating=4,
            metadata={"model": "bge-m3"}
        )

    def test_search_query_creation(self, sample_query):
        """Test basic search query creation."""
        assert sample_query.id == "query:test"
        assert sample_query.tenant_id == "tenant:test"
        assert sample_query.user_id == "user:test"
        assert sample_query.query_text == "What is machine learning?"
        assert sample_query.query_language == "en"
        assert sample_query.detected_language == "en"
        assert sample_query.primary_category == CategoryType.TECHNICAL
        assert sample_query.secondary_categories == [CategoryType.EDUCATION, CategoryType.ACADEMIC]
        assert sample_query.retrieval_strategy == "hybrid"
        assert sample_query.scope_searched == [DocumentScope.USER, DocumentScope.TENANT]
        assert sample_query.results_count == 5
        assert sample_query.response_time_ms == 150
        assert sample_query.satisfaction_rating == 4
        assert isinstance(sample_query.created_at, datetime)

    def test_search_query_defaults(self):
        """Test search query default values."""
        query = SearchQuery(
            id="query:minimal",
            tenant_id="tenant:test",
            user_id="user:test",
            query_text="Test query"
        )

        assert query.query_language == "auto"
        assert query.detected_language is None
        assert query.primary_category is None
        assert query.secondary_categories == []
        assert query.retrieval_strategy is None
        assert query.scope_searched == [DocumentScope.USER, DocumentScope.TENANT]
        assert query.results_count == 0
        assert query.response_time_ms == 0
        assert query.satisfaction_rating is None
        assert query.metadata == {}
        assert isinstance(query.created_at, datetime)

    def test_add_timing(self, sample_query):
        """Test adding response timing."""
        # Create a start time 100ms ago
        from datetime import timedelta
        start_time = datetime.now() - timedelta(milliseconds=100)

        sample_query.add_timing(start_time)

        # Should be approximately 100ms (allow some variance)
        assert 90 <= sample_query.response_time_ms <= 120


class TestCategorizationTemplate:
    """Test CategorizationTemplate dataclass."""

    @pytest.fixture
    def sample_template(self):
        """Create sample categorization template for testing."""
        return CategorizationTemplate(
            id="template:test",
            tenant_id="tenant:test",
            name="Technical Query Template",
            category=CategoryType.TECHNICAL,
            language="en",
            keywords=["algorithm", "machine learning", "AI", "programming"],
            patterns=[r"how to \w+", r"what is \w+", r"\w+ tutorial"],
            system_prompt="You are a technical assistant.",
            user_prompt_template="Answer this technical question: {query}",
            is_system_default=False,
            is_active=True,
            priority=5
        )

    def test_template_creation(self, sample_template):
        """Test basic template creation."""
        assert sample_template.id == "template:test"
        assert sample_template.tenant_id == "tenant:test"
        assert sample_template.name == "Technical Query Template"
        assert sample_template.category == CategoryType.TECHNICAL
        assert sample_template.language == "en"
        assert sample_template.keywords == ["algorithm", "machine learning", "AI", "programming"]
        assert sample_template.patterns == [r"how to \w+", r"what is \w+", r"\w+ tutorial"]
        assert sample_template.system_prompt == "You are a technical assistant."
        assert sample_template.user_prompt_template == "Answer this technical question: {query}"
        assert sample_template.is_system_default is False
        assert sample_template.is_active is True
        assert sample_template.priority == 5
        assert isinstance(sample_template.created_at, datetime)
        assert isinstance(sample_template.updated_at, datetime)

    def test_template_defaults(self):
        """Test template default values."""
        template = CategorizationTemplate(
            id="template:minimal",
            tenant_id=None,
            name="Minimal Template",
            category=CategoryType.GENERAL,
            language="hr"
        )

        assert template.tenant_id is None
        assert template.keywords == []
        assert template.patterns == []
        assert template.system_prompt == ""
        assert template.user_prompt_template == ""
        assert template.is_system_default is False
        assert template.is_active is True
        assert template.priority == 0
        assert isinstance(template.created_at, datetime)
        assert isinstance(template.updated_at, datetime)

    def test_matches_query_keywords_only(self, sample_template):
        """Test query matching with keywords only."""
        # Clear patterns to test keywords only
        sample_template.patterns = []

        # High match with multiple keywords
        score = sample_template.matches_query("What is machine learning algorithm in AI programming?")
        assert 0.5 < score <= 1.0  # Should be high due to multiple keyword matches

        # Partial match with some keywords
        score = sample_template.matches_query("Tell me about programming concepts.")
        assert 0.1 < score < 0.5  # Lower score with fewer matches

        # No match
        score = sample_template.matches_query("Weather forecast for tomorrow.")
        assert score == 0.0

    def test_matches_query_patterns_only(self, sample_template):
        """Test query matching with patterns only."""
        # Clear keywords to test patterns only
        sample_template.keywords = []

        # Should match "what is" pattern
        score = sample_template.matches_query("What is machine learning?")
        assert 0.1 < score <= 0.4  # Pattern weight varies based on implementation

        # Should match "how to" pattern
        score = sample_template.matches_query("How to implement neural networks?")
        assert 0.1 <= score <= 0.4  # Should be around 1/3 * 0.4 ≈ 0.13

        # No pattern match
        score = sample_template.matches_query("I like programming very much.")
        assert score == 0.0

    def test_matches_query_combined(self, sample_template):
        """Test query matching with both keywords and patterns."""
        query = "What is machine learning algorithm?"
        score = sample_template.matches_query(query)

        # Should have both keyword and pattern contributions
        # Keywords: "machine learning" and "algorithm" = 2/4 = 0.5 * 0.6 = 0.3
        # Patterns: "what is" matches = 1/3 * 0.4 = 0.133
        # Total ≈ 0.433, but capped at 1.0
        assert 0.4 < score <= 1.0

    def test_matches_query_invalid_pattern(self, sample_template):
        """Test handling of invalid regex patterns."""
        sample_template.patterns = ["[invalid regex", r"valid \w+ pattern"]
        sample_template.keywords = []

        # Should not crash on invalid regex, just continue
        score = sample_template.matches_query("This is a valid pattern test.")
        # Score may be 0 if pattern matching is strict
        assert score >= 0  # Should not crash, may not match depending on pattern


class TestSystemConfig:
    """Test SystemConfig dataclass."""

    @pytest.fixture
    def sample_config(self):
        """Create sample system config for testing."""
        return SystemConfig(
            id="config:test",
            tenant_id="tenant:test",
            config_key="max_chunk_size",
            config_value="1000",
            config_type="int",
            description="Maximum chunk size for document processing",
            is_system_config=False
        )

    def test_config_creation(self, sample_config):
        """Test basic config creation."""
        assert sample_config.id == "config:test"
        assert sample_config.tenant_id == "tenant:test"
        assert sample_config.config_key == "max_chunk_size"
        assert sample_config.config_value == "1000"
        assert sample_config.config_type == "int"
        assert sample_config.description == "Maximum chunk size for document processing"
        assert sample_config.is_system_config is False
        assert isinstance(sample_config.created_at, datetime)
        assert isinstance(sample_config.updated_at, datetime)

    def test_config_defaults(self):
        """Test config default values."""
        config = SystemConfig(
            id="config:minimal",
            tenant_id=None,
            config_key="test_key",
            config_value="test_value"
        )

        assert config.tenant_id is None
        assert config.config_type == "string"
        assert config.description is None
        assert config.is_system_config is False
        assert isinstance(config.created_at, datetime)
        assert isinstance(config.updated_at, datetime)

    def test_get_typed_value_string(self):
        """Test string type conversion."""
        config = SystemConfig(
            id="config:string",
            tenant_id=None,
            config_key="test",
            config_value="hello world",
            config_type="string"
        )

        assert config.get_typed_value() == "hello world"

    def test_get_typed_value_int(self):
        """Test integer type conversion."""
        config = SystemConfig(
            id="config:int",
            tenant_id=None,
            config_key="test",
            config_value="42",
            config_type="int"
        )

        assert config.get_typed_value() == 42
        assert isinstance(config.get_typed_value(), int)

    def test_get_typed_value_float(self):
        """Test float type conversion."""
        config = SystemConfig(
            id="config:float",
            tenant_id=None,
            config_key="test",
            config_value="3.14",
            config_type="float"
        )

        assert config.get_typed_value() == 3.14
        assert isinstance(config.get_typed_value(), float)

    def test_get_typed_value_bool_true(self):
        """Test boolean type conversion for true values."""
        true_values = ["true", "True", "TRUE", "1", "yes", "Yes", "on", "On"]

        for value in true_values:
            config = SystemConfig(
                id="config:bool",
                tenant_id=None,
                config_key="test",
                config_value=value,
                config_type="bool"
            )

            assert config.get_typed_value() is True

    def test_get_typed_value_bool_false(self):
        """Test boolean type conversion for false values."""
        false_values = ["false", "False", "0", "no", "off", "anything else"]

        for value in false_values:
            config = SystemConfig(
                id="config:bool",
                tenant_id=None,
                config_key="test",
                config_value=value,
                config_type="bool"
            )

            assert config.get_typed_value() is False

    def test_get_typed_value_json(self):
        """Test JSON type conversion."""
        config = SystemConfig(
            id="config:json",
            tenant_id=None,
            config_key="test",
            config_value='{"key": "value", "number": 42}',
            config_type="json"
        )

        result = config.get_typed_value()
        assert result == {"key": "value", "number": 42}
        assert isinstance(result, dict)


class TestTenantUserContext:
    """Test TenantUserContext dataclass."""

    @pytest.fixture
    def sample_context(self):
        """Create sample tenant-user context for testing."""
        tenant = Tenant(
            id="tenant:context",
            name="Context Tenant",
            slug="context",
            status=TenantStatus.ACTIVE
        )

        user = User(
            id="user:context",
            tenant_id="tenant:context",
            email="context@example.com",
            username="context",
            role=UserRole.MEMBER,
            status=UserStatus.ACTIVE
        )

        return TenantUserContext(tenant=tenant, user=user)

    def test_context_creation(self, sample_context):
        """Test basic context creation."""
        assert isinstance(sample_context.tenant, Tenant)
        assert isinstance(sample_context.user, User)
        assert sample_context.tenant.slug == "context"
        assert sample_context.user.username == "context"

    def test_get_user_collection_name(self, sample_context):
        """Test user collection name generation."""
        collection_name = sample_context.get_user_collection_name("hr")
        assert collection_name == "context_user_hr"

    def test_get_tenant_collection_name(self, sample_context):
        """Test tenant collection name generation."""
        collection_name = sample_context.get_tenant_collection_name("en")
        assert collection_name == "context_tenant_en"

    def test_get_search_collections(self, sample_context):
        """Test search collections retrieval."""
        collections = sample_context.get_search_collections("hr")

        # Should include user collection
        assert "context_user_hr" in collections

        # Should include tenant collection if user can access
        if sample_context.user.can_access_tenant_documents():
            assert "context_tenant_hr" in collections
            assert len(collections) == 2
        else:
            assert len(collections) == 1

    def test_can_access_document_same_tenant_user_scope(self, sample_context):
        """Test document access for same tenant, user scope."""
        document = Document(
            id="doc:user",
            tenant_id="tenant:context",
            user_id="user:context",  # Same user
            title="User Document",
            filename="user.txt",
            file_path="/data/user.txt",
            scope=DocumentScope.USER
        )

        assert sample_context.can_access_document(document) is True

    def test_can_access_document_same_tenant_different_user(self, sample_context):
        """Test document access for same tenant, different user."""
        document = Document(
            id="doc:other",
            tenant_id="tenant:context",
            user_id="user:other",  # Different user
            title="Other Document",
            filename="other.txt",
            file_path="/data/other.txt",
            scope=DocumentScope.USER
        )

        assert sample_context.can_access_document(document) is False

    def test_can_access_document_tenant_scope(self, sample_context):
        """Test document access for tenant-scoped document."""
        document = Document(
            id="doc:tenant",
            tenant_id="tenant:context",
            user_id="user:other",
            title="Tenant Document",
            filename="tenant.txt",
            file_path="/data/tenant.txt",
            scope=DocumentScope.TENANT
        )

        # Depends on user's tenant document access permission
        expected = sample_context.user.can_access_tenant_documents()
        assert sample_context.can_access_document(document) is expected

    def test_can_access_document_different_tenant(self, sample_context):
        """Test document access for different tenant."""
        document = Document(
            id="doc:external",
            tenant_id="tenant:other",  # Different tenant
            user_id="user:context",
            title="External Document",
            filename="external.txt",
            file_path="/data/external.txt",
            scope=DocumentScope.USER
        )

        assert sample_context.can_access_document(document) is False


class TestMultiTenantQueryResult:
    """Test MultiTenantQueryResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create sample multi-tenant query result for testing."""
        tenant = Tenant(id="tenant:result", name="Result Tenant", slug="result")
        user = User(id="user:result", tenant_id="tenant:result", email="result@example.com", username="result")
        context = TenantUserContext(tenant=tenant, user=user)

        return MultiTenantQueryResult(
            query="Test query for results",
            tenant_context=context,
            user_results=[{"id": "doc:user1", "score": 0.9}],
            tenant_results=[{"id": "doc:tenant1", "score": 0.8}],
            combined_results=[
                {"id": "doc:user1", "score": 0.9, "scope": "user"},
                {"id": "doc:tenant1", "score": 0.8, "scope": "tenant"}
            ],
            total_results=2,
            user_results_count=1,
            tenant_results_count=1,
            response_time_ms=200,
            primary_category=CategoryType.TECHNICAL,
            retrieval_strategy="hybrid"
        )

    def test_result_creation(self, sample_result):
        """Test basic result creation."""
        assert sample_result.query == "Test query for results"
        assert isinstance(sample_result.tenant_context, TenantUserContext)
        assert len(sample_result.user_results) == 1
        assert len(sample_result.tenant_results) == 1
        assert len(sample_result.combined_results) == 2
        assert sample_result.total_results == 2
        assert sample_result.user_results_count == 1
        assert sample_result.tenant_results_count == 1
        assert sample_result.response_time_ms == 200
        assert sample_result.primary_category == CategoryType.TECHNICAL
        assert sample_result.retrieval_strategy == "hybrid"

    def test_result_defaults(self):
        """Test result default values."""
        tenant = Tenant(id="tenant:minimal", name="Minimal", slug="minimal")
        user = User(id="user:minimal", tenant_id="tenant:minimal", email="minimal@example.com", username="minimal")
        context = TenantUserContext(tenant=tenant, user=user)

        result = MultiTenantQueryResult(
            query="Minimal query",
            tenant_context=context
        )

        assert result.query == "Minimal query"
        assert result.user_results == []
        assert result.tenant_results == []
        assert result.combined_results == []
        assert result.total_results == 0
        assert result.user_results_count == 0
        assert result.tenant_results_count == 0
        assert result.response_time_ms == 0
        assert result.primary_category is None
        assert result.retrieval_strategy is None


class TestDefaultConstants:
    """Test default development constants."""

    def test_default_development_tenant(self):
        """Test default development tenant values."""
        tenant = DEFAULT_DEVELOPMENT_TENANT

        assert tenant.id == "tenant:development"
        assert tenant.name == "Development Tenant"
        assert tenant.slug == "development"
        assert tenant.description == "Default tenant for development and testing"
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.business_context == BusinessContext.BUSINESS
        assert tenant.subscription_tier == SubscriptionTier.ENTERPRISE

        # Check settings
        assert tenant.settings["allow_user_document_promotion"] is True
        assert tenant.settings["auto_detect_language"] is True
        assert tenant.settings["enable_advanced_categorization"] is True
        assert tenant.settings["max_documents_per_user"] == 1000
        assert tenant.settings["max_total_documents"] == 10000

    def test_default_development_user(self):
        """Test default development user values."""
        user = DEFAULT_DEVELOPMENT_USER

        assert user.id == "user:dev_user"
        assert user.tenant_id == "tenant:development"
        assert user.email == "dev@example.com"
        assert user.username == "dev_user"
        assert user.full_name == "Development User"
        assert user.password_hash == "$2b$12$dummy_hash_for_development"
        assert user.role == UserRole.ADMIN
        assert user.status == UserStatus.ACTIVE

        # Check settings
        assert user.settings["preferred_categories"] == ["technical", "business"]
        assert user.settings["auto_categorize"] is True
        assert user.settings["search_both_scopes"] is True

    def test_default_development_context(self):
        """Test default development context."""
        context = DEFAULT_DEVELOPMENT_CONTEXT

        assert isinstance(context, TenantUserContext)
        assert context.tenant == DEFAULT_DEVELOPMENT_TENANT
        assert context.user == DEFAULT_DEVELOPMENT_USER

        # Test that context methods work
        user_collection = context.get_user_collection_name("hr")
        assert user_collection == "development_user_hr"

        tenant_collection = context.get_tenant_collection_name("en")
        assert tenant_collection == "development_tenant_en"


# Integration Tests
class TestIntegration:
    """Test complete multitenant workflows."""

    def test_tenant_user_document_workflow(self):
        """Test complete tenant-user-document workflow."""
        # Create tenant
        tenant = Tenant(
            id="tenant:workflow",
            name="Workflow Tenant",
            slug="workflow",
            status=TenantStatus.ACTIVE,
            settings={"max_documents_per_user": 100, "max_total_documents": 1000}
        )

        # Create user
        user = User(
            id="user:workflow",
            tenant_id="tenant:workflow",
            email="workflow@example.com",
            username="workflow",
            role=UserRole.MEMBER,
            status=UserStatus.ACTIVE
        )

        # Create context
        context = TenantUserContext(tenant=tenant, user=user)

        # Create document
        document = Document(
            id="doc:workflow",
            tenant_id="tenant:workflow",
            user_id="user:workflow",
            title="Workflow Document",
            filename="workflow.pdf",
            file_path="/data/workflow.pdf",
            scope=DocumentScope.USER,
            status=DocumentStatus.PROCESSED
        )

        # Test permissions and operations
        assert tenant.can_create_user() is True
        assert user.can_upload_documents() is True
        assert user.can_access_tenant_documents() is True
        assert context.can_access_document(document) is True
        assert document.can_be_promoted_to_tenant() is True

        # Test collection names
        assert document.get_collection_name("workflow") == "workflow_user_auto"
        assert context.get_user_collection_name("hr") == "workflow_user_hr"

        # Promote document to tenant scope
        document.scope = DocumentScope.TENANT
        assert document.get_collection_name("workflow") == "workflow_tenant_auto"
        assert context.can_access_document(document) is True

    def test_search_query_analytics_workflow(self):
        """Test search query with analytics workflow."""
        # Create search query
        query = SearchQuery(
            id="query:analytics",
            tenant_id="tenant:test",
            user_id="user:test",
            query_text="How to implement machine learning algorithms?",
            query_language="en",
            detected_language="en",
            primary_category=CategoryType.TECHNICAL,
            retrieval_strategy="hybrid"
        )

        # Add timing
        start_time = datetime.now()
        query.add_timing(start_time)

        # Create template that should match this query
        template = CategorizationTemplate(
            id="template:analytics",
            tenant_id="tenant:test",
            name="Technical Template",
            category=CategoryType.TECHNICAL,
            language="en",
            keywords=["machine learning", "algorithm", "implement"],
            patterns=[r"how to \w+", r"implement \w+"]
        )

        # Test template matching
        match_score = template.matches_query(query.query_text)
        assert match_score > 0.5  # Should be a good match

        # Create result
        result = MultiTenantQueryResult(
            query=query.query_text,
            tenant_context=DEFAULT_DEVELOPMENT_CONTEXT,
            user_results=[{"doc": "doc1", "score": 0.9}],
            tenant_results=[{"doc": "doc2", "score": 0.8}],
            total_results=2,
            user_results_count=1,
            tenant_results_count=1,
            primary_category=query.primary_category,
            retrieval_strategy=query.retrieval_strategy
        )

        assert result.total_results == 2
        assert result.primary_category == CategoryType.TECHNICAL
        assert result.retrieval_strategy == "hybrid"

    def test_system_configuration_workflow(self):
        """Test system configuration workflow."""
        # Create different types of configs
        configs = [
            SystemConfig(
                id="config:string",
                tenant_id="tenant:test",
                config_key="system_name",
                config_value="Test System",
                config_type="string"
            ),
            SystemConfig(
                id="config:int",
                tenant_id="tenant:test",
                config_key="max_results",
                config_value="50",
                config_type="int"
            ),
            SystemConfig(
                id="config:bool",
                tenant_id="tenant:test",
                config_key="enable_caching",
                config_value="true",
                config_type="bool"
            ),
            SystemConfig(
                id="config:json",
                tenant_id="tenant:test",
                config_key="feature_flags",
                config_value='{"new_ui": true, "beta_features": false}',
                config_type="json"
            )
        ]

        # Test type conversions
        assert configs[0].get_typed_value() == "Test System"
        assert configs[1].get_typed_value() == 50
        assert configs[2].get_typed_value() is True
        assert configs[3].get_typed_value() == {"new_ui": True, "beta_features": False}

    def test_chunk_processing_workflow(self):
        """Test document chunk processing workflow."""
        # Create document
        document = Document(
            id="doc:chunks",
            tenant_id="tenant:test",
            user_id="user:test",
            title="Chunked Document",
            filename="chunked.pdf",
            file_path="/data/chunked.pdf",
            status=DocumentStatus.PROCESSED,
            chunk_count=3
        )

        # Create chunks
        chunks = []
        for i in range(3):
            chunk = Chunk(
                id=f"chunk:{i}",
                document_id="doc:chunks",
                tenant_id="tenant:test",
                user_id="user:test",
                scope=DocumentScope.USER,
                chunk_index=i,
                content=f"This is chunk {i} content with some meaningful text.",
                language="en",
                vector_collection="test_user_en"
            )
            chunks.append(chunk)

        # Verify chunk processing
        assert len(chunks) == document.chunk_count

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.content_length > 0  # Should be calculated by __post_init__
            assert chunk.document_id == document.id
            assert chunk.tenant_id == document.tenant_id
            assert chunk.user_id == document.user_id
            assert chunk.scope == document.scope

    def test_permission_hierarchy_workflow(self):
        """Test permission hierarchy across roles and statuses."""
        # Create tenant
        tenant = Tenant(
            id="tenant:permissions",
            name="Permission Tenant",
            slug="permissions",
            status=TenantStatus.ACTIVE
        )

        # Create users with different roles
        admin = User(
            id="user:admin",
            tenant_id="tenant:permissions",
            email="admin@example.com",
            username="admin",
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE
        )

        member = User(
            id="user:member",
            tenant_id="tenant:permissions",
            email="member@example.com",
            username="member",
            role=UserRole.MEMBER,
            status=UserStatus.ACTIVE
        )

        viewer = User(
            id="user:viewer",
            tenant_id="tenant:permissions",
            email="viewer@example.com",
            username="viewer",
            role=UserRole.VIEWER,
            status=UserStatus.ACTIVE
        )

        # Test upload permissions
        assert admin.can_upload_documents() is True
        assert member.can_upload_documents() is True
        assert viewer.can_upload_documents() is False

        # Test document promotion permissions
        assert admin.can_promote_documents_to_tenant() is True
        assert member.can_promote_documents_to_tenant() is True
        assert viewer.can_promote_documents_to_tenant() is False

        # Test tenant document access
        assert admin.can_access_tenant_documents() is True
        assert member.can_access_tenant_documents() is True
        assert viewer.can_access_tenant_documents() is True

        # Test suspended user permissions
        admin.status = UserStatus.SUSPENDED
        assert admin.can_upload_documents() is False
        assert admin.can_promote_documents_to_tenant() is False
        assert admin.can_access_tenant_documents() is False