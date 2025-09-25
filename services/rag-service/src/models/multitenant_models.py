"""
Multi-tenant data models for RAG system.

These models define the structure for tenant-user hierarchy and document scoping.
Corresponds to the SurrealDB schema in schema/multitenant_schema.surql
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, cast

from src.retrieval.categorization import CategoryType

from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_data_transformation,
    log_performance_metric,
)


class TenantStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"


class UserRole(Enum):
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class UserStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"


class DocumentScope(Enum):
    USER = "user"  # Private to user
    TENANT = "tenant"  # Shared within tenant


class DocumentStatus(Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"


class FileType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "md"
    HTML = "html"


class Language(Enum):
    AUTO = "auto"
    MULTILINGUAL = "multilingual"


class BusinessContext(Enum):
    BUSINESS = "business"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    GOVERNMENT = "government"


class SubscriptionTier(Enum):
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class Tenant:
    """Represents a tenant in the multi-tenant RAG system."""

    id: str
    name: str
    slug: str
    description: str | None = None
    status: TenantStatus = TenantStatus.ACTIVE
    settings: dict[str, Any] = field(default_factory=dict)
    business_context: BusinessContext = BusinessContext.BUSINESS
    subscription_tier: SubscriptionTier = SubscriptionTier.BASIC
    language_preference: Language = Language.AUTO
    cultural_context: BusinessContext = BusinessContext.BUSINESS
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_supported_languages(self) -> list[str]:
        """Get supported languages from configuration."""
        from src.utils.config_loader import get_supported_languages

        return get_supported_languages()

    def get_collection_name(self, scope: DocumentScope, language: str) -> str:
        """Generate ChromaDB collection name for this tenant."""
        collection_name = f"{self.slug}_{scope.value}_{language}"

        # Handle Weaviate collection name capitalization for features tenant
        # Weaviate auto-capitalizes the first letter when creating collections
        if self.slug == "features" and scope == DocumentScope.USER:
            collection_name = f"Features_{scope.value}_{language}"

        return collection_name

    def can_create_user(self) -> bool:
        """Check if tenant can create new users."""
        return self.status == TenantStatus.ACTIVE

    def get_max_documents(self) -> int:
        """Get maximum documents allowed for this tenant."""
        return cast(int, self.settings["max_total_documents"])

    def get_max_documents_per_user(self) -> int:
        """Get maximum documents per user for this tenant."""
        return cast(int, self.settings["max_documents_per_user"])


@dataclass
class User:
    """Represents a user within a tenant."""

    id: str
    tenant_id: str
    email: str
    username: str
    full_name: str | None = None
    password_hash: str = ""
    role: UserRole = UserRole.MEMBER
    status: UserStatus = UserStatus.ACTIVE
    settings: dict[str, Any] = field(default_factory=dict)
    language_preference: Language = Language.AUTO
    last_login_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_preferred_languages(self) -> list[str]:
        """Get user's preferred languages, fallback to system supported."""
        user_prefs = self.settings["preferred_languages"]
        if user_prefs:
            return cast(list[str], user_prefs)

        # Fallback to system default
        from src.utils.config_loader import get_shared_config

        config = get_shared_config()
        if "default" not in config["languages"]:
            raise ValueError("Missing 'default' in languages configuration")
        default_lang = config["languages"]["default"]
        return [default_lang]

    def can_upload_documents(self) -> bool:
        """Check if user can upload documents."""
        return self.status == UserStatus.ACTIVE and self.role != UserRole.VIEWER

    def can_access_tenant_documents(self) -> bool:
        """Check if user can access tenant-scoped documents."""
        return self.status == UserStatus.ACTIVE

    def can_promote_documents_to_tenant(self) -> bool:
        """Check if user can promote documents to tenant scope."""
        return self.status == UserStatus.ACTIVE and self.role in [UserRole.ADMIN, UserRole.MEMBER]

    def get_preferred_categories(self) -> list[str]:
        """Get user's preferred document categories."""
        return cast(list[str], self.settings["preferred_categories"])


@dataclass
class Document:
    """Represents a document in the RAG system."""

    id: str
    tenant_id: str
    user_id: str
    title: str
    filename: str
    file_path: str
    file_size: int = 0
    file_type: FileType = FileType.TXT
    language: str = "auto"
    scope: DocumentScope = DocumentScope.USER
    status: DocumentStatus = DocumentStatus.UPLOADED
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    chunk_count: int = 0
    content: str = ""
    source_path: str | None = None
    mime_type: str | None = None
    checksum: str | None = None
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_collection_name(self, tenant_slug: str) -> str:
        """Generate ChromaDB collection name for this document."""
        return f"{tenant_slug}_{self.scope.value}_{self.language}"

    def is_processed(self) -> bool:
        """Check if document has been processed."""
        return self.status == DocumentStatus.PROCESSED

    def can_be_promoted_to_tenant(self) -> bool:
        """Check if document can be promoted to tenant scope."""
        return self.scope == DocumentScope.USER and self.status == DocumentStatus.PROCESSED

    def get_display_name(self) -> str:
        """Get human-readable display name."""
        return self.title if self.title else self.filename


@dataclass
class Chunk:
    """Represents a processed document chunk with vector metadata."""

    id: str
    document_id: str
    tenant_id: str
    user_id: str
    scope: DocumentScope
    chunk_index: int
    content: str
    content_length: int = 0
    language: str = "auto"
    embedding_model: str = "bge-m3"
    vector_collection: str = ""
    vector_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    embedding_dimension: int = 768
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Set computed fields after initialization."""
        if not self.content_length:
            self.content_length = len(self.content)


@dataclass
class SearchQuery:
    """Represents a search query with analytics."""

    id: str
    tenant_id: str
    user_id: str
    query_text: str
    query_language: str = "auto"
    detected_language: str | None = None
    primary_category: CategoryType | None = None
    secondary_categories: list[CategoryType] = field(default_factory=list)
    retrieval_strategy: str | None = None  # String strategy name
    scope_searched: list[DocumentScope] = field(default_factory=lambda: [DocumentScope.USER, DocumentScope.TENANT])
    results_count: int = 0
    response_time_ms: int = 0
    satisfaction_rating: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_timing(self, start_time: datetime):
        """Add response timing to query."""
        self.response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)


@dataclass
class CategorizationTemplate:
    """Template for category-specific prompts and patterns."""

    id: str
    tenant_id: str | None
    name: str
    category: CategoryType
    language: str
    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    system_prompt: str = ""
    user_prompt_template: str = ""
    is_system_default: bool = False
    is_active: bool = True
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def matches_query(self, query: str) -> float:
        """Calculate match score for query against this template."""
        import re

        logger = get_system_logger()
        log_component_start(
            "query_template",
            "matches_query",
            query_length=len(query),
            keywords_count=len(self.keywords),
            patterns_count=len(self.patterns),
        )

        query_lower = query.lower()
        score = 0.0

        # Check keyword matches
        keyword_matches = sum(1 for keyword in self.keywords if keyword.lower() in query_lower)
        keyword_score = 0.0
        if self.keywords:
            keyword_score = (keyword_matches / len(self.keywords)) * 0.6
            score += keyword_score

        log_performance_metric("query_template", "matches_query", "keyword_matches", keyword_matches)
        log_performance_metric("query_template", "matches_query", "keyword_score", keyword_score)

        # Check pattern matches
        pattern_matches = 0
        for pattern in self.patterns:
            try:
                if re.search(pattern.lower(), query_lower):
                    pattern_matches += 1
            except re.error:
                continue

        pattern_score = 0.0
        if self.patterns:
            pattern_score = (pattern_matches / len(self.patterns)) * 0.4
            score += pattern_score

        log_performance_metric("query_template", "matches_query", "pattern_matches", pattern_matches)
        log_performance_metric("query_template", "matches_query", "pattern_score", pattern_score)

        final_score = min(score, 1.0)
        log_performance_metric("query_template", "matches_query", "final_score", final_score)

        log_data_transformation(
            "query_template",
            "score_calculation",
            f"Input: query '{query[:30]}...' vs template",
            f"Score calculated: {final_score:.3f} (keywords: {keyword_score:.3f}, patterns: {pattern_score:.3f})",
            keyword_matches=keyword_matches,
            pattern_matches=pattern_matches,
            final_score=final_score,
        )

        logger.debug(
            "query_template",
            "matches_query",
            f"Template matching: {keyword_matches}/{len(self.keywords)} keywords, {pattern_matches}/{len(self.patterns)} patterns, score: {final_score:.3f}",
        )

        log_component_end(
            "query_template",
            "matches_query",
            f"Match score calculated: {final_score:.3f}",
            final_score=final_score,
            keyword_contribution=keyword_score,
            pattern_contribution=pattern_score,
        )

        return final_score


@dataclass
class SystemConfig:
    """System configuration key-value pairs."""

    id: str
    tenant_id: str | None
    config_key: str
    config_value: str
    config_type: str = "string"
    description: str | None = None
    is_system_config: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_typed_value(self) -> Any:
        """Return config value with appropriate type conversion."""
        if self.config_type == "int":
            return int(self.config_value)
        elif self.config_type == "float":
            return float(self.config_value)
        elif self.config_type == "bool":
            return self.config_value.lower() in ("true", "1", "yes", "on")
        elif self.config_type == "json":
            import json

            return json.loads(self.config_value)
        else:
            return self.config_value


# ===============================================
# Multi-Tenant Context and Scoping
# ===============================================


@dataclass
class TenantUserContext:
    """Context for multi-tenant operations."""

    tenant: Tenant
    user: User

    def get_user_collection_name(self, language: str) -> str:
        """Get ChromaDB collection name for user documents."""
        return f"{self.tenant.slug}_user_{language}"

    def get_tenant_collection_name(self, language: str) -> str:
        """Get ChromaDB collection name for tenant documents."""
        return f"{self.tenant.slug}_tenant_{language}"

    def get_search_collections(self, language: str) -> list[str]:
        """Get all collections user should search in."""
        collections = []

        # Always include user's private documents
        collections.append(self.get_user_collection_name(language))

        # Include tenant documents if user can access them
        if self.user.can_access_tenant_documents():
            collections.append(self.get_tenant_collection_name(language))

        return collections

    def can_access_document(self, document: Document) -> bool:
        """Check if user can access a specific document."""
        # Must be same tenant
        if document.tenant_id != self.tenant.id:
            return False

        # Can access tenant documents if user has permission
        if document.scope == DocumentScope.TENANT:
            return self.user.can_access_tenant_documents()

        # Can access own user documents
        if document.scope == DocumentScope.USER:
            return document.user_id == self.user.id

        return False


@dataclass
class MultiTenantQueryResult:
    """Result from multi-tenant query with scope information."""

    query: str
    tenant_context: TenantUserContext
    user_results: list[dict[str, Any]] = field(default_factory=list)
    tenant_results: list[dict[str, Any]] = field(default_factory=list)
    combined_results: list[dict[str, Any]] = field(default_factory=list)
    total_results: int = 0
    user_results_count: int = 0
    tenant_results_count: int = 0
    response_time_ms: int = 0
    primary_category: CategoryType | None = None
    retrieval_strategy: str | None = None  # String strategy name


# ===============================================
# Default Instances for Development
# ===============================================

DEFAULT_DEVELOPMENT_TENANT = Tenant(
    id="tenant:development",
    name="Development Tenant",
    slug="development",
    description="Default tenant for development and testing",
    status=TenantStatus.ACTIVE,
    business_context=BusinessContext.BUSINESS,
    subscription_tier=SubscriptionTier.ENTERPRISE,
    language_preference=Language.AUTO,
    cultural_context=BusinessContext.BUSINESS,
    settings={
        "allow_user_document_promotion": True,
        "auto_detect_language": True,
        "enable_advanced_categorization": True,
        "max_documents_per_user": 1000,
        "max_total_documents": 10000,
    },
)

DEFAULT_DEVELOPMENT_USER = User(
    id="user:dev_user",
    tenant_id="tenant:development",
    email="dev@example.com",
    username="dev_user",
    full_name="Development User",
    password_hash="$2b$12$dummy_hash_for_development",
    role=UserRole.ADMIN,
    status=UserStatus.ACTIVE,
    language_preference=Language.AUTO,
    settings={"preferred_categories": ["technical", "business"], "auto_categorize": True, "search_both_scopes": True},
)

DEFAULT_DEVELOPMENT_CONTEXT = TenantUserContext(tenant=DEFAULT_DEVELOPMENT_TENANT, user=DEFAULT_DEVELOPMENT_USER)
